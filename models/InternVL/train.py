import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator

from models.internvl_wrapper import load_internvl_model
from data.dataset import GazeDataset
from losses import get_loss_fn
from utils import set_seed, get_logger, setup_distributed, is_main_process, cleanup_distributed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='internvl/internvl2-8b')
    parser.add_argument('--gaze_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--loss_fn', type=str, default='kl', choices=['kl', 'mse', 'focal'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    rank = setup_distributed()
    logger = get_logger("train")

    accelerator = Accelerator()
    device = accelerator.device

    model = load_internvl_model(args.model_name_or_path)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model = accelerator.prepare(model)

    dataset = GazeDataset(args.image_path, args.gaze_path, args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    dataloader = accelerator.prepare(dataloader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = get_loss_fn(args.loss_fn)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            images, gazes = batch
            images, gazes = images.to(device), gazes.to(device)
            outputs = model.forward_with_attention(images, max_length=args.max_length, return_attn=True)
            attention_maps = outputs['attn']
            loss = loss_fn(attention_maps, gazes)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * args.gradient_accumulation_steps
        if is_main_process() and (epoch + 1) % args.save_every == 0:
            accelerator.unwrap_model(model).save_pretrained(os.path.join(args.output_dir, f"checkpoint-{epoch+1}"))
        if is_main_process():
            logger.info(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss:.4f}")

    if is_main_process():
        accelerator.unwrap_model(model).save_pretrained(args.output_dir)
    cleanup_distributed()

if __name__ == "__main__":
    main()
