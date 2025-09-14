import os
import random
import logging
import torch
import numpy as np
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank
    else:
        return 0

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def get_logger(name="train"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        formatter = logging.Formatter(fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def normalize_attention(attn_map):
    attn_map = attn_map.view(attn_map.size(0), -1)
    attn_map = attn_map / (attn_map.sum(dim=-1, keepdim=True) + 1e-8)
    return attn_map

def resize_and_normalize_gaze(gaze_tensor, target_size):
    gaze_resized = F.interpolate(gaze_tensor.unsqueeze(1), size=target_size, mode='bilinear', align_corners=False)
    gaze_resized = gaze_resized.squeeze(1)
    gaze_resized = gaze_resized / (gaze_resized.sum(dim=(1, 2), keepdim=True) + 1e-8)
    return gaze_resized

def preprocess_image(pil_image, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(pil_image)
