import torch
from torch import nn
from transformers import AutoTokenizer, AutoProcessor, AutoModel, BitsAndBytesConfig
from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig, TaskType

def load_internvl_model(model_name_or_path, qlora=True):
    """Load and return an InternVL model wrapped for training."""
    return InternVLWrapper(model_name_or_path, qlora=qlora)

class InternVLWrapper(nn.Module):
    def __init__(self, model_name="OpenGVLab/InternVL2-8B", qlora=True):
        super().__init__()
        self.model_name = model_name
        self.qlora = qlora

        if qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModel.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            model = prepare_model_for_kbit_training(model)
            peft_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.model = get_peft_model(model, peft_config)
        else:
            self.model = AutoModel.from_pretrained(
                model_name, 
                torch_dtype=torch.bfloat16, 
                device_map="auto",
                trust_remote_code=True,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True,
            return_dict=True
        )
        return outputs

    def get_cross_attentions(self, pixel_values, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True
            )
        all_attentions = outputs.cross_attentions
        first_12_layers = all_attentions[:12]
        avg_layer_attn = [torch.stack(layer).mean(dim=0) for layer in first_12_layers]
        attn_map = torch.stack(avg_layer_attn).mean(dim=0)
        return attn_map

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=32):
        generation_output = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=False,
            output_scores=False
        )
        return generation_output

    def tokenize(self, question):
        tokenized = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64
        )
        return tokenized

    def process_image(self, image):
        processed = self.processor(images=image, return_tensors="pt")
        return processed["pixel_values"]

    def forward_with_attention(self, images, max_length=32, return_attn=True):
        """Forward pass that returns attention maps for gaze supervision."""
        # Handle both PIL images and tensors
        if isinstance(images, torch.Tensor):
            # Convert tensor back to PIL for processing (temporary workaround)
            # In practice, you'd want to modify the processing pipeline
            pixel_values = images
        else:
            # Create a simple question for the image
            question = "What do you see in this image?"
            
            # Process the image and question
            pixel_values = self.process_image(images)
            
        # Use a default question for attention extraction
        question = "What do you see in this image?"
        tokenized = self.tokenize(question)
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Ensure tensors are on the same device
        if hasattr(self.model, 'device'):
            device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        
        # Get cross-attention maps
        attn_map = self.get_cross_attentions(pixel_values, input_ids, attention_mask)
        
        # Return in the format expected by the training code
        if return_attn:
            return {"attn": attn_map}
        else:
            return {"attn": None}
