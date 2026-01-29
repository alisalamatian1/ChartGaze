import torch
import math
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
                target_modules=["wqkv", "wo", "w1", "w2", "w3"],
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
        
        # Get the underlying model (unwrap PEFT if needed)
        base_model = self.model.base_model.model if hasattr(self.model, 'base_model') else self.model
        
        # Store important model attributes
        self.num_image_token = base_model.num_image_token
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.img_start_token = '<img>'
        self.img_end_token = '</img>'
        
    def _get_base_model(self):
        """Get the underlying InternVLChatModel (unwrapping PEFT if needed)."""
        if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'model'):
            return self.model.base_model.model
        return self.model
    
    def _set_img_context_token_id(self):
        """Set img_context_token_id on the base model."""
        base_model = self._get_base_model()
        base_model.img_context_token_id = self.img_context_token_id

    def prepare_inputs(self, images, questions, num_patches_list=None):
        """
        Prepare inputs in the format expected by InternVL2.
        
        Args:
            images: PIL Image or list of PIL Images
            questions: str or list of str
            num_patches_list: Number of image patches per image (default: 1 per image)
        
        Returns:
            dict with pixel_values, input_ids, attention_mask, image_flags
        """
        from PIL import Image
        
        # Handle single image/question
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(images, Image.Image):
            images = [images]
        
        if num_patches_list is None:
            num_patches_list = [1] * len(images)
        
        base_model = self._get_base_model()
        
        # Process images
        pixel_values_list = []
        for img in images:
            # Use the model's built-in image processing if available
            if hasattr(base_model, 'vision_model'):
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((448, 448)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                pixel_values_list.append(transform(img))
            else:
                raise ValueError("Cannot find vision preprocessing method")
        
        pixel_values = torch.stack(pixel_values_list)
        
        # Prepare text with image tokens
        processed_questions = []
        for idx, (question, num_patches) in enumerate(zip(questions, num_patches_list)):
            # Add <image> placeholder if not present
            if '<image>' not in question:
                question = '<image>\n' + question
            
            # Replace <image> with actual image tokens
            image_tokens = (
                self.img_start_token + 
                '<IMG_CONTEXT>' * self.num_image_token * num_patches + 
                self.img_end_token
            )
            question = question.replace('<image>', image_tokens, 1)
            processed_questions.append(question)
        
        # Tokenize
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(
            processed_questions, 
            return_tensors='pt', 
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Create image_flags
        batch_size = len(images)
        image_flags = torch.ones((batch_size, 1), dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'image_flags': image_flags,
        }

    def forward(self, pixel_values, input_ids, attention_mask, labels=None, image_flags=None):
        """Forward pass through the model."""
        # Create default image_flags if not provided
        if image_flags is None:
            batch_size = pixel_values.shape[0] if pixel_values is not None else input_ids.shape[0]
            image_flags = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        
        self._set_img_context_token_id()
        
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_flags=image_flags,
            labels=labels,
            output_attentions=True,
            return_dict=True
        )
        return outputs

    def extract_vision_attention(self, attentions, input_ids):
        """
        Extract attention weights from text tokens to vision tokens.
        
        InternVL2 merges vision embeddings into the input sequence at positions
        marked by IMG_CONTEXT tokens. This extracts the attention from text
        tokens to those vision token positions.
        
        Args:
            attentions: Tuple of attention tensors from each layer
                       Each tensor has shape (batch, num_heads, seq_len, seq_len)
            input_ids: Input token IDs to locate vision token positions
        
        Returns:
            Attention map of shape (batch, num_vision_tokens) normalized
        """
        # Find positions of image context tokens
        img_token_mask = (input_ids == self.img_context_token_id)
        
        # Get attention from first N layers
        num_layers_to_use = min(12, len(attentions))
        selected_attentions = attentions[:num_layers_to_use]
        
        batch_size = input_ids.shape[0]
        attn_maps = []
        
        for b in range(batch_size):
            # Get vision token positions for this batch item
            vision_positions = img_token_mask[b].nonzero(as_tuple=True)[0]
            
            if len(vision_positions) == 0:
                # No vision tokens found, return zeros
                attn_maps.append(torch.zeros(self.num_image_token, device=input_ids.device))
                continue
            
            # Get non-vision (text) token positions
            text_positions = (~img_token_mask[b]).nonzero(as_tuple=True)[0]
            
            # Collect attention from text to vision across layers
            layer_attns = []
            for layer_attn in selected_attentions:
                # layer_attn shape: (batch, heads, seq, seq)
                # Extract attention from text positions to vision positions
                # Average over heads
                attn = layer_attn[b].mean(dim=0)  # (seq, seq)
                
                # Get attention from text tokens to vision tokens
                # Shape: (num_text_tokens, num_vision_tokens)
                text_to_vision = attn[text_positions][:, vision_positions]
                
                # Average over text tokens to get (num_vision_tokens,)
                avg_attn = text_to_vision.mean(dim=0)
                layer_attns.append(avg_attn)
            
            # Average across layers
            combined_attn = torch.stack(layer_attns).mean(dim=0)
            attn_maps.append(combined_attn)
        
        # Stack batch
        attn_maps = torch.stack(attn_maps)  # (batch, num_vision_tokens)
        
        return attn_maps

    def get_attention_map(self, pixel_values, input_ids, attention_mask, image_flags=None):
        """
        Get attention map from text to vision tokens.
        
        Returns:
            Attention map reshaped to 2D grid (batch, H, W)
        """
        if image_flags is None:
            batch_size = pixel_values.shape[0]
            image_flags = torch.ones((batch_size, 1), dtype=torch.long, device=input_ids.device)
        
        self._set_img_context_token_id()
        
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                output_attentions=True,
                return_dict=True
            )
        
        # Extract attention to vision tokens
        attentions = outputs.attentions
        attn_map = self.extract_vision_attention(attentions, input_ids)
        
        # Reshape to 2D grid
        grid_size = int(math.sqrt(attn_map.shape[-1]))
        if grid_size * grid_size != attn_map.shape[-1]:
            # Pad if not perfect square
            target_size = grid_size * grid_size
            if attn_map.shape[-1] < target_size:
                padding = torch.zeros(
                    attn_map.shape[0], 
                    target_size - attn_map.shape[-1],
                    device=attn_map.device
                )
                attn_map = torch.cat([attn_map, padding], dim=-1)
            else:
                attn_map = attn_map[:, :target_size]
        
        attn_map = attn_map.view(attn_map.shape[0], grid_size, grid_size)
        
        return attn_map

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=32):
        """Generate text given image and prompt."""
        self._set_img_context_token_id()
            
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
    
    def chat(self, image, question, max_new_tokens=512):
        """
        Simple chat interface matching InternVL2's expected usage.
        
        Args:
            image: PIL Image
            question: str
            max_new_tokens: int
        
        Returns:
            Generated response string
        """
        base_model = self._get_base_model()
        self._set_img_context_token_id()
        
        # Use the model's built-in chat method if available
        if hasattr(base_model, 'chat'):
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((448, 448)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pixel_values = transform(image).unsqueeze(0)
            device = next(self.model.parameters()).device
            pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
            
            generation_config = {
                'max_new_tokens': max_new_tokens,
                'do_sample': False,
            }
            
            response = base_model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config
            )
            return response
        else:
            # Fallback to manual generation
            inputs = self.prepare_inputs(image, question)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.generate(
                inputs['pixel_values'].to(torch.bfloat16),
                inputs['input_ids'],
                inputs['attention_mask'],
                max_new_tokens=max_new_tokens
            )
            
            response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            return response
