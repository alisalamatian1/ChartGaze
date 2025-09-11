import json, os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random

torch.seed(1970)
np.seed(1970)
random.seed(1970)

# --- CONFIG ---
VAL_JSON     = "./data/val/data.json"
IMAGE_FOLDER = "./data/val"
MODEL_PATH   = "./models/ChartGemma/model"
NP_OUTPUT = "./models/ChartGemma/model/results/vis/"
OUTPUT_JSONL = "./models/ChartGemma/model/results/eval_chartgemma.jsonl"
ATTN_OUT_DIR = "./models/ChartGemma/model/results/vis/val/"
NUM_VISION_TOKENS = 1024
ATTN_COLLECT_FIRST_N_LAYERS = 10
NUM_ATTN_MAPS_TO_SAVE = 50
NUM_ATTN_IMAGES_TO_PICTURE = 10

def collect_attention_maps(outputs, start_idx, end_idx, skip_K = 0):
    # breakpoint()
    attns = outputs.attentions
    attn_maps_all_layers = []
    # only looking at the first output token because the question is supposedly T/F and the rest of the generated tokens are not interesting right now
    for out_token_idx in range(len(attns)):
        for i, attn_layer in enumerate(attns[out_token_idx][:ATTN_COLLECT_FIRST_N_LAYERS]):
            attn_layer_avg_heads = attn_layer.squeeze(0).mean(dim=0)   
            # take out the query -> vision part 
            # the first token is seq-len * seq-len because it includes the input token, the rest are 1*seq-len because of next token prediction
            if attn_layer_avg_heads.shape[0] >= end_idx:
                layer_attn_avg_head_vis = attn_layer_avg_heads[end_idx:, start_idx:end_idx]
            else:
                layer_attn_avg_head_vis = attn_layer_avg_heads[:, start_idx:end_idx]
            # print(f"layer_attn_avg_head_vis shape before mean across text query: {layer_attn_avg_head_vis.shape}")
            # take an average over the text query tokens
            layer_attn_avg_head_vis = layer_attn_avg_head_vis.mean(dim=0, keepdim=True)
            # print(f"layer_attn_avg_head_vis shape after mean across text query: {layer_attn_avg_head_vis.shape}")
            grid_size = math.ceil(math.sqrt(layer_attn_avg_head_vis.shape[-1]))
            # append 0 in the front if the number of tokens is not a perfect square
            # this can happen when using different vision token strategies
            if grid_size * grid_size != layer_attn_avg_head_vis.shape[-1]:
                # print(f"grid size square: {grid_size * grid_size} vs attn_avg_head_vis: {layer_attn_avg_head_vis.shape[-1]}")
                zero_pad = torch.zeros((layer_attn_avg_head_vis.shape[0], grid_size * grid_size - layer_attn_avg_head_vis.shape[-1]), dtype=layer_attn_avg_head_vis.dtype, device=layer_attn_avg_head_vis.device)
                layer_attn_avg_head_vis = torch.cat((zero_pad, layer_attn_avg_head_vis), dim=-1)
                

            # print(f"the dimension for layer {i} before taking a mean: {layer_attn_avg_head_vis.shape}")
                
            attn_maps_all_layers.append(layer_attn_avg_head_vis)

    mean_attn_map = torch.stack(attn_maps_all_layers, dim=0).mean(dim=0)    
    print(f"mean attention map shape: {mean_attn_map.shape}")  
    _, topk_indices = torch.topk(mean_attn_map, skip_K, dim=-1)
    # remove elements with top K for each batch
    mean_attn_map_skipped = mean_attn_map.scatter(dim=-1, index=topk_indices, value=0)  
    max_attn_map = mean_attn_map_skipped / mean_attn_map_skipped.max(1, keepdim=True)[0]        
    # print(f"max attention map shape: {max_attn_map.shape}")    
    
    # reshape attention map to grid_size*grid_size
    final_attn_map = max_attn_map.view(grid_size, grid_size)
    return final_attn_map, grid_size

def visualize_attn_map(attn_maps, chart_base_image, count, np_path, output_path = "evaluation", image_size=448, figure_size=8, alpha=0.5):
    """
    Visualize attention maps with proper sizing and formatting.
    
    Parameters:
    - attn_maps: Tensor containing attention map data
    - output_path: Path where the visualization will be saved
    - image_size: Target size for the attention map (square)
    - figure_size: Base size for the output figure in inches
    
    Returns:
    - None (saves figure to disk)
    """
    # Resize attention map to match image dimensions (square)
    attn_resized = F.interpolate(
        attn_maps.unsqueeze(0).unsqueeze(0),  # Add batch + channel dims
        size=(image_size, image_size),         # Target size as tuple (H, W)
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()  # Remove added dims and convert to numpy

    # save the np file
    if count < NUM_ATTN_MAPS_TO_SAVE:
        np.save(os.path.join(NP_OUTPUT, np_path), attn_resized)

    if count < NUM_ATTN_IMAGES_TO_PICTURE:
        chart_np = np.array(chart_base_image.resize((image_size, image_size)))
        
        # Create figure with appropriate sizing
        fig, ax = plt.subplots(figsize=(figure_size, figure_size), dpi=100)

        ax.imshow(chart_np)
        
        # Display attention map with a heat colormap
        im = ax.imshow(attn_resized, cmap='hot', alpha=alpha)
        
        # Add colorbar with automatic sizing
        fig.colorbar(im, ax=ax)
        
        # Remove axes for cleaner visualization
        ax.axis("off")
        
        # Save the figure
        fig.tight_layout()
        fig.savefig(ATTN_OUT_DIR)
        fig.savefig(os.path.join(ATTN_OUT_DIR, output_path))
        
        # Close the figure to free memory
        plt.close(fig)

# --- SETUP ---
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = PaliGemmaForConditionalGeneration.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

os.makedirs(ATTN_OUT_DIR, exist_ok=True)

# --- EVALUATION LOOP ---
with open(VAL_JSON, "r") as f, open(OUTPUT_JSONL, "w") as outf:
    val_data = json.load(f)
    for idx, item in enumerate(val_data):
        image = Image.open(os.path.join(IMAGE_FOLDER, item["image"])).convert("RGB")
        prompt = item["conversations"][0]["value"][len("<image>\n"):]
        # if you want to explicitly ask it to answer yes/no:
        prompt =  prompt + "Please answer with Yes or No."
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        for k, v in inputs.items():
            print(k, v.dtype, v.device)

        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.inference_mode():
            max_id = inputs["input_ids"].max().item()
            assert max_id < model.config.vocab_size, (
                f"Found token ID {max_id} ≥ vocab size {model.config.vocab_size}"
            )

            out = model.generate(
                input_ids=      inputs["input_ids"],
                attention_mask= inputs["attention_mask"],
                pixel_values=   inputs["pixel_values"],

                pad_token_id=   processor.tokenizer.pad_token_id,
                eos_token_id=   processor.tokenizer.eos_token_id,
                bos_token_id=   processor.tokenizer.bos_token_id,

                num_beams=      4,
                max_new_tokens= 512,
                output_attentions=True,
                return_dict_in_generate=True,
            )


        # breakpoint()
        # --- Decode Response ---
        gen_ids = out.sequences[:, prompt_len:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        # --- Save Output ---
        outf.write(json.dumps({
            "question_id": item["id"],
            "prompt": prompt,
            "text": text
        }) + "\n")

        image_token_indecies = (inputs["input_ids"] == model.config.image_token_index).nonzero(as_tuple=True)[1]
        start_idx = image_token_indecies[0]
        end_idx = image_token_indecies[-1]

        final_attn_map, grid_size = collect_attention_maps(out, start_idx, end_idx)

        visualize_attn_map(final_attn_map, image, idx,  item["attn_map"].split("attn_maps/")[1], f"q{idx}_llm_vis_avg_tokens")


