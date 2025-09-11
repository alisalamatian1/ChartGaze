import torch
from torch.utils.data import Dataset
from typing import Any, Dict
import random
from PIL import Image, ImageFile
from io import BytesIO
import json
import os
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ChartGemmaDataset(Dataset):
    """
    data_args contain:
    data_path : where the data.json is 
    image_parent_folder
    guided_attn_map: boolean for whether to use the attn loss or not
    """

    def __init__(
        self,
        data_args,
    ):
        super().__init__()
        self.data_args = data_args
        self.list_data_dict = json.load(open(self.data_args["data_path"], "r"))
        self.dataset_length = len(self.list_data_dict)

    def __len__(self) -> int:
        return self.dataset_length
    
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

    def __getitem__(self, idx: int) -> Dict:
        sources = self.list_data_dict[idx]
        guided_attn_map_square = None
        image_file = sources['image']
        image_parent_folder = self.data_args["image_parent_folder"]
        image = Image.open(os.path.join(image_parent_folder, image_file)).convert('RGB')
        img_width_orig, img_height_orig = image.size
        if self.data_args["guided_attn_map"]:
            attn_map_file = sources['attn_map']
            guided_attn_map_orig = np.load(os.path.join(image_parent_folder, attn_map_file))
            guided_attn_map_orig = torch.tensor(guided_attn_map_orig)
            guided_attn_map_square = self.center_tensor_on_square(guided_attn_map_orig)
            guided_attn_map_square = torch.nn.functional.interpolate(guided_attn_map_square.unsqueeze(0).unsqueeze(0), size=image.size[-2:], mode='bilinear', align_corners=False).squeeze(0).squeeze(0)

        prompt = sources["conversations"][0]["value"][len("<image>\n"):]
        expected_answer = sources["conversations"][1]["value"]
        return image, prompt, expected_answer, guided_attn_map_square