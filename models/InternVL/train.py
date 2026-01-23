import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from accelerate import Accelerator

from models.internvl_wrapper import load_internvl_model
from data.dataset import InternVLGazeDataset
from losses import get_loss_fn, normalize_attention_map, normalize_gaze_map
from utils import set_seed, get_logger, setup_distributed, is_main_process, cleanup_distributed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='OpenGVLab/InternVL2-8B')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with images/, gaze_maps/, and metadata.json')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--loss_fn', type=str, default='wmse', choices=['kl', 'mse', 'wmse', 'focal'])
    parser.add_argument('--loss_alpha', type=float, default=1.1, help='Alpha parameter for weighted MSE loss')
    parser.add_argument('--skip_top_k', type=int, default=5, help='Number of top-K attention values to skip during normalization')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--qlora', action='store_true', default=True, help='Use QLoRA quantization')
    parser.add_argument('--lm_loss_weight', type=float, default=1.0, help='Weight for language modeling loss')
    parser.add_argument('--attn_loss_weight', type=float, default=1.0, help='Weight for attention/gaze loss')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    rank = setup_distributed()
    logger = get_logger("train")

    accelerator = Accelerator()
    device = accelerator.device

    # Load model with QLoRA already configured in the wrapper
    logger.info(f"Loading model from {args.model_name_or_path}...")
    model = load_internvl_model(args.model_name_or_path, qlora=args.qlora)
    
    # Prepare dataset - use InternVLGazeDataset which includes questions/answers
    logger.info(f"Loading dataset from {args.data_dir}...")
    dataset = InternVLGazeDataset(
        data_dir=args.data_dir, 
        processor=None,  # We'll handle processing in the collate function
        max_length=args.max_length
    )
    
    def collate_fn(batch):
        """Custom collate function to prepare batch for InternVL2."""
        images = []
        questions = []
        answers = []
        gaze_maps = []
        
        for item in batch:
            # Load image
            img = Image.open(item['image_path']).convert('RGB')
            images.append(img)
            questions.append(item['question'])
            answers.append(item['answer'])
            
            # Load and process gaze map
            gaze = Image.open(item['gaze_path']).convert('L')
            gaze_array = np.array(gaze, dtype=np.float32) / 255.0
            gaze_tensor = torch.from_numpy(gaze_array).unsqueeze(0)  # Add channel dim: (1, H, W)
            gaze_maps.append(gaze_tensor)
        
        # Prepare inputs using the model's method
        inputs = model.prepare_inputs(images, questions)
        
        # Resize gaze maps to match attention map size
        # InternVL2's attention grid size depends on num_image_token
        grid_size = int(math.sqrt(model.num_image_token))
        gaze_tensors = torch.stack(gaze_maps)
        gaze_tensors = F.interpolate(
            gaze_tensors, 
            size=(grid_size, grid_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        return {
            'pixel_values': inputs['pixel_values'],
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'image_flags': inputs['image_flags'],
            'gaze_maps': gaze_tensors,
            'answers': answers,
        }
    
    # Override dataset __getitem__ to return paths instead of processed data
    class PathDataset(torch.utils.data.Dataset):
        def __init__(self, base_dataset):
            self.data = base_dataset.data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    path_dataset = PathDataset(dataset)
    dataloader = DataLoader(
        path_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with model
        collate_fn=collate_fn
    )
    
    # Prepare for distributed training
    model.model, dataloader = accelerator.prepare(model.model, dataloader)

    # Optimizer - only train LoRA parameters
    trainable_params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    optimizer = accelerator.prepare(optimizer)
    
    # Loss function with normalization
    attn_loss_fn = get_loss_fn(args.loss_fn, alpha=args.loss_alpha)
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"Loss function: {args.loss_fn}, Alpha: {args.loss_alpha}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    model.model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        total_lm_loss = 0
        total_attn_loss = 0
        
        for step, batch in enumerate(dataloader):
            # Move tensors to device
            pixel_values = batch['pixel_values'].to(device, dtype=torch.bfloat16)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            image_flags = batch['image_flags'].to(device)
            gaze_maps = batch['gaze_maps'].to(device)
            
            # Forward pass with attention
            model._set_img_context_token_id()
            outputs = model.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                output_attentions=True,
                return_dict=True
            )
            
            # Language modeling loss (optional, if labels are provided)
            lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)
            
            # Extract attention maps and compute gaze supervision loss
            attentions = outputs.attentions
            attn_map = model.extract_vision_attention(attentions, input_ids)
            
            # Reshape attention map to 2D grid
            grid_size = int(math.sqrt(attn_map.shape[-1]))
            attn_map_2d = attn_map.view(attn_map.shape[0], grid_size, grid_size)
            
            # Normalize both maps before computing loss
            gaze_normalized = normalize_gaze_map(gaze_maps.squeeze(1))  # Remove channel dim
            attn_normalized = normalize_attention_map(attn_map_2d, skip_top_k=args.skip_top_k)
            
            # Compute attention loss
            attn_loss = attn_loss_fn(attn_normalized, gaze_normalized)
            
            # Combined loss
            loss = args.lm_loss_weight * lm_loss + args.attn_loss_weight * attn_loss
            loss = loss / args.gradient_accumulation_steps
            
            accelerator.backward(loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
            total_lm_loss += lm_loss.item()
            total_attn_loss += attn_loss.item()
            
            if step % 10 == 0 and is_main_process():
                logger.info(f"Epoch {epoch+1}, Step {step}: Loss={loss.item():.4f}, LM={lm_loss.item():.4f}, Attn={attn_loss.item():.4f}")
        
        # Save checkpoint
        if is_main_process() and (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            accelerator.unwrap_model(model.model).save_pretrained(save_path)
            logger.info(f"Saved checkpoint to {save_path}")
        
        if is_main_process():
            avg_loss = total_loss / len(dataloader)
            avg_lm_loss = total_lm_loss / len(dataloader)
            avg_attn_loss = total_attn_loss / len(dataloader)
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Total Loss: {avg_loss:.4f}, LM Loss: {avg_lm_loss:.4f}, Attn Loss: {avg_attn_loss:.4f}")

    # Save final model
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        accelerator.unwrap_model(model.model).save_pretrained(args.output_dir)
        model.tokenizer.save_pretrained(args.output_dir)
        logger.info(f"Saved final model to {args.output_dir}")
    
    cleanup_distributed()

if __name__ == "__main__":
    main()
