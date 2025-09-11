from PIL import Image
import requests
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
import torch
import json
import os
from chartgemma_dataset import ChartGemmaDataset
import lightning as L
# from finetune import ChartGemmaModelPLModule
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

class ChartGemmaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model
        self.batch_size = config.get("batch_size")

MODEL_PATH = "ahmed-masry/chartgemma"
USE_QLORA = False
USE_LORA = True
IMAGE_TOKEN_INDEX = 257152
SKIP_K = 0

def get_config():
    return {
        "max_epochs": 7,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 8,
        "lr": 1e-4,
        "batch_size": 1,
        "seed": 1970,
        "num_nodes": 1,
        "warmup_steps": 50,
        "result_path": "./result",
        "verbose": True,
    }

def get_processor():
    return AutoProcessor.from_pretrained(MODEL_PATH)

def get_model(processor):
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
    elif USE_LORA:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
        )
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
        )
    return model


VAL_JSON = "<path to downloaded json file>/val/data.json"
IMAGE_FOLDER = "<path to downloaded image>/val/images"
OUTPUT_JSONL = "chartgemma_attn_refined_response.json"

# --- SETUP ---
model_path = "ahmed-masry/chartgemma"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(model_path)
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# if we want to load the model from a checkpoint
# config = get_config()
# processor = get_processor()
# model = get_model(processor)
# device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model module from checkpoint
# pl_module = ChartGemmaModelPLModule.load_from_checkpoint(
#     "/home/alisalam/scratch/ChartGemmaFinetuning/model/checkpoints_both_loss_S001_6layers_seed42_lr5e5/epepoch=06.ckpt",
#     config=config,
#     processor=processor,
#     model=model,
    
# )
# pl_module.to(device)

# --- EVALUATION LOOP ---
with open(VAL_JSON, "r") as f, open(OUTPUT_JSONL, "w") as outf:
    val_data = json.load(f)
    for idx, item in enumerate(val_data):
        image = Image.open(os.path.join(IMAGE_FOLDER, item["image"].split("images/")[1])).convert("RGB")
        prompt = item["conversations"][0]["value"] + " Explain your reasoning."
        inputs = processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            out = model.generate(
                **inputs, num_beams=4, max_new_tokens=512
            )

        # --- Decode Response ---
        gen_ids = out[:, prompt_len:]
        text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        print(text)
        # --- Save Output ---
        outf.write(json.dumps({
            "question_id": item["attn_map"].split("attn_maps/")[1].split(".npy")[0],
            "prompt": prompt,
            "text": text
        }) + "\n")