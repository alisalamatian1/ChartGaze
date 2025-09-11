from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from chartgemma_dataset import ChartGemmaDataset
import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
import math
import torch.nn.functional as F
import random
import os
from pytorch_lightning import seed_everything

IMAGE_TOKEN_INDEX = 257152
USE_LORA = False
USE_QLORA = True
SKIP_K = 0
model_path = "ChartGemma/model"
SEED = 2025

def make_deterministic(seed=2025):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CUDNN: deterministic algorithms, no auto-tuner
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

    # Enforce PyTorch to throw if a nondeterministic op is used
    torch.use_deterministic_algorithms(True)

    # If using Lightning, also seed it (will seed workers, PRNGs, etc.)
    seed_everything(seed, workers=True)

make_deterministic(SEED)


processor = AutoProcessor.from_pretrained(model_path)

## Load model

# Three options for training, from the lowest precision training to the highest precision training:
# - QLora
# - Standard Lora
# - Full fine-tuning
if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
elif USE_LORA:
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )
else:
    # for full fine-tuning, we can speed up the model using Flash Attention
    # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        _attn_implementation="flash_attention_2",
    )
    for param in model.vision_tower.parameters():
       param.requires_grad = False

    for param in model.multi_modal_projector.parameters():
       param.requires_grad = False

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


lora_config = LoraConfig(
    r=32,
    lora_alpha = 64,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

train_data_args = {
    "data_path": "data/train/data.json", 
    "image_parent_folder": "data/train",
    "guided_attn_map": True
}

val_data_args = {
    "data_path": "data/val/data.json", 
    "image_parent_folder": "data/val",
    "guided_attn_map": True
}

train_dataset = ChartGemmaDataset(train_data_args)
val_dataset = ChartGemmaDataset(val_data_args)

def train_collate_fn(examples):
    images = []
    input_texts = []
    output_texts = []
    gt_attn_maps = []

    for example in examples:
        image, input_text, output_text, gt_attn_map = example
        images.append(image)
        input_texts.append(input_text)
        output_texts.append(output_text)
        gt_attn_maps.append(gt_attn_map)

    inputs = processor(text=input_texts, images=images, suffix=output_texts, return_tensors = "pt", padding=True,
                       truncation="only_second", tokenize_newline_separately=False)
    
    input_ids = inputs["input_ids"]
    token_type_ids = inputs["token_type_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]
    labels = inputs["labels"]

    return input_ids, token_type_ids, attention_mask, pixel_values, labels, gt_attn_maps

def eval_collate_fn(examples):
    # we only feed the prompt to the model
    images = []
    texts = []
    answers = []
    gt_attn_maps = []

    for example in examples:
        image, text, answer, gt_attn_map = example
        images.append(image)
        texts.append(text)
        answers.append(answer)
        gt_attn_maps.append(gt_attn_map)

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, tokenize_newline_separately=False)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    pixel_values = inputs["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers, gt_attn_maps


class ChartGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")

    def collect_attention_maps(self, outputs, start_idx, end_idx, n_layer_to_collect=10):
        # collecting attention map (detail: avg all heads, all layers, and all tokens)
        attn_maps_all_layers = []
        for layer_idx, layer_attn in enumerate(outputs.attentions[:n_layer_to_collect]):
            # layer_attn shape: (batch_size, num_heads, seq_len, seq_len)
            # the seq_len, seq_len part is representing the softmax(QK^T)
            layer_attn_avg_head = layer_attn.mean(dim=1)
            # layer_attn_avg_head = layer_attn.max(dim=1)[0]
            # fixed attn query -> vision: batch_size, text_token_section_size, image_token_section_size
            layer_attn_avg_head_vis = layer_attn_avg_head[:, end_idx+1:, start_idx:end_idx+1]
            # averaging over all the queries: batch_size, image_token_section_size
            layer_attn_avg_head_vis = layer_attn_avg_head_vis.mean(dim=1)
            grid_size = math.ceil(math.sqrt(layer_attn_avg_head_vis.shape[-1]))
            # append 0 in the front if the number of tokens is not a perfect square
            # this can happen when using different vision token strategies
            if grid_size * grid_size != layer_attn_avg_head_vis.shape[-1]:
                zero_pad = torch.zeros((layer_attn_avg_head_vis.shape[0], grid_size * grid_size - layer_attn_avg_head_vis.shape[-1]), dtype=layer_attn_avg_head_vis.dtype, device=layer_attn_avg_head_vis.device)
                layer_attn_avg_head_vis = torch.cat((zero_pad, layer_attn_avg_head_vis), dim=-1)

            attn_maps_all_layers.append(layer_attn_avg_head_vis)

        # avg all layers: shape batch_size, padded seq len
        attn_maps_all_layers = torch.stack(attn_maps_all_layers, dim=0).mean(dim=0)
        return attn_maps_all_layers, grid_size
    
    def weighted_mse_loss(self, input, target, alpha=1.1):
        # note that target is at max 1, so this operation is safe to do
        w = 1 / (alpha - target)
        w_mse = torch.sum(w * (target - input)**2)
        return w_mse
    
    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, pixel_values, labels, g_attn_maps = batch

        outputs = self.model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                pixel_values=pixel_values,
                                labels=labels,
                                output_attentions=True)
        loss = outputs.loss

        print(f"lan loss: {loss}")

        '''
        Steps to calculate the attention loss
        1: extract the attention map 
        2: calculate the loss
        '''
        image_token_indecies = (input_ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[1]
        start_idx = image_token_indecies[0]
        end_idx = image_token_indecies[-1]

        final_attn_map, grid_size = self.collect_attention_maps(outputs, start_idx, end_idx)

        # process the ground truth attention map
        g_attn_maps = g_attn_maps / g_attn_maps.max()
        g_attn_maps = g_attn_maps.unsqueeze(1)
        g_attn_maps = F.interpolate(g_attn_maps, size=(grid_size, grid_size), mode='bilinear', align_corners=False)
        g_attn_maps = g_attn_maps.squeeze(1).view(-1, grid_size * grid_size)

        # process the model attention map and remove elements with top K for each batch
        _, topk_indices = torch.topk(final_attn_map, SKIP_K, dim=-1)
        final_attn_map = final_attn_map.scatter(dim=-1, index=topk_indices, value=0)
        final_attn_map = final_attn_map / final_attn_map.max(1, keepdim=True)[0]

        loss_attn_maps = self.weighted_mse_loss(final_attn_map, g_attn_maps)

        if self.training:
            print(f"Training Attention Loss {loss_attn_maps}")
        else:
            print(f"Evaluation Attention Loss {loss_attn_maps}")
        loss += loss_attn_maps
        print(f"total loss: {loss}")

        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get("lr"))
        return optimizer
    
    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(SEED)
        return DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=2, generator=g, worker_init_fn=lambda wid: np.random.seed(SEED + wid))
    
    def val_dataloader(self):
        g = torch.Generator()
        g.manual_seed(SEED)
        return DataLoader(val_dataset, collate_fn=eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=2, generator=g, worker_init_fn=lambda wid: np.random.seed(SEED + wid))


config = {"max_epochs": 2,
          # "val_check_interval": 0.2, # how many times we want to validate during an epoch
          "check_val_every_n_epoch": 1,
          "gradient_clip_val": 1.0,
          "accumulate_grad_batches": 8,
          "lr": 1e-4,
          "batch_size": 4,
          "seed":2025,
          "num_nodes": 1,
          "warmup_steps": 50,
          "result_path": "./result",
          "verbose": True,
}

model_module = ChartGemmaModelPLModule(config, processor, model)

trainer = L.Trainer(
        accelerator="gpu",
        devices=[0],
        max_epochs=config.get("max_epochs"),
        accumulate_grad_batches=config.get("accumulate_grad_batches"),
        check_val_every_n_epoch=config.get("check_val_every_n_epoch"),
        gradient_clip_val=config.get("gradient_clip_val"),
        precision="16-mixed",
        num_sanity_val_steps=0,
        deterministic=True
        # logger=wandb_logger,
)

trainer.fit(model_module)

save_path = "ChartGemma/model/finetuned_model"
os.makedirs(save_path, exist_ok=True)

model_module.model.save_pretrained(save_path)
model_module.processor.save_pretrained(save_path)
