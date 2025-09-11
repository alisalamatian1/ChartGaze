import copy
from dataclasses import dataclass
import json
from typing import Dict,  Sequence, TYPE_CHECKING
from PIL import Image, ImageFile
import numpy as np
import os

from .text_preprocess import TextPreprocess
from .image_preprocess import ImagePreprocess
from ..utils.arguments import DataArguments
from ..utils.constants import *


import transformers
import torch
from torch.utils.data import Dataset



ImageFile.LOAD_TRUNCATED_IMAGES = True

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.text_preprocess = TextPreprocess(tokenizer, data_args.conv_version)
        self.image_preprocess = ImagePreprocess(data_args.image_processor, data_args)

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def center_tensor_on_square(self, tensor):
        h, w = tensor.shape
        size = max(h, w)  # Determine the square size
        
        # Create a square zero tensor
        square_tensor = torch.zeros((size, size), dtype=tensor.dtype, device=tensor.device)
        
        # Compute start indices to center the tensor
        start_h = (size - h) // 2
        start_w = (size - w) // 2
        
        # Paste the original tensor in the center
        square_tensor[start_h:start_h + h, start_w:start_w + w] = tensor
        
        return square_tensor
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        data_dict = self.text_preprocess(copy.deepcopy(sources["conversations"]))
        if 'image' in sources:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            img_width_orig, img_height_orig = image.size
            image = self.image_preprocess(image)
            data_dict['image'] = image
            if self.data_args.guided_attn_map:
                attn_map_file = self.list_data_dict[i]['attn_map']
                # guided_attn_folder = os.path.join(image_folder, 'attn_maps')
                # if not os.path.exists(guided_attn_folder):
                #     raise ValueError(f'Guided attention map folder does not exist: {guided_attn_folder}. Please put the guided attention map in the folder with the name of attn_maps.')
                # guided_attn_map_orig = np.load(os.path.join(guided_attn_folder, image_file.split('/')[-1].replace('.png', '.npy')))
                guided_attn_map_orig = np.load(os.path.join(image_folder, attn_map_file))
                guided_attn_map_orig = torch.tensor(guided_attn_map_orig)

                guided_attn_map_square = self.center_tensor_on_square(guided_attn_map_orig)

                guided_attn_map_square = torch.nn.functional.interpolate(guided_attn_map_square.unsqueeze(0).unsqueeze(0), size=image.shape[-2:], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
                # make sure sum to 1
                # guided_attn_map_square = guided_attn_map_square / guided_attn_map_square.sum()
                data_dict['g_attn_map'] = guided_attn_map_square

        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # print(f'{i}:{sources}')
            crop_size = getattr(self.data_args.image_processor, 'crop_size', getattr(self.data_args.image_processor, 'size'))
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == self.tokenizer.eos_token_id] = -300
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        labels = labels[:, :self.tokenizer.model_max_length]
        # FIXME: This is a hack for handling phi and stablelm, as they have the same eos, pad and unk. We want the model
        # FIXME: to predict the eos in the input ids, but we also use the id of eos to pad sequence, so we use a temp
        # FIXME: eos id first, and convert them back.
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            for input_id in input_ids:
                input_id[input_id == -300] = self.tokenizer.eos_token_id

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

            if 'g_attn_map' in instances[0]:
                g_attn_maps = [instance['g_attn_map'] for instance in instances]
                if all(x is not None and x.shape == g_attn_maps[0].shape for x in g_attn_maps):
                    batch['g_attn_maps'] = torch.stack(g_attn_maps)
                else:
                    batch['g_attn_maps'] = g_attn_maps

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    eval_dataset = None
    if data_args.validation_image_folder:
        data_args.image_folder = data_args.validation_image_folder
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.validation_data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)
