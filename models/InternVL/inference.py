import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from models.internvl_wrapper import load_internvl_model
from utils import preprocess_image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image for inference')
    parser.add_argument('--output_dir', type=str, default='./inference_outputs',
                       help='Directory to save inference outputs')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Size to resize input image')
    parser.add_argument('--question', type=str, default="What do you see in this image?",
                       help='Question to ask about the image')
    parser.add_argument('--save_attention', action='store_true',
                       help='Save attention map visualization')
    parser.add_argument('--generate_text', action='store_true',
                       help='Generate text response')
    return parser.parse_args()


def load_trained_model(model_path, base_model_name="OpenGVLab/InternVL2-8B"):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    # Load base model
    model = load_internvl_model(base_model_name, qlora=True)
    
    # Load trained weights
    if os.path.isdir(model_path):
        # If it's a directory, load the PEFT adapter
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, model_path)
    else:
        # If it's a single file, load state dict
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    model.eval()
    return model


def preprocess_input_image(image_path, image_size=224):
    """Load and preprocess image for inference."""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    # Preprocess for model
    processed_image = preprocess_image(image, image_size)
    
    return image, processed_image, original_size


def extract_attention_map(model, image_tensor, question="What do you see in this image?"):
    """Extract attention map from the model."""
    with torch.no_grad():
        # Get attention maps
        outputs = model.forward_with_attention(image_tensor.unsqueeze(0), return_attn=True)
        attention_map = outputs['attn']
        
        # Process attention map
        if len(attention_map.shape) > 2:
            # Average over heads and layers if needed
            while len(attention_map.shape) > 2:
                attention_map = attention_map.mean(dim=1)
        
        # Normalize attention map
        attention_map = attention_map.squeeze()
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
        
    return attention_map


def generate_text_response(model, image_tensor, question="What do you see in this image?"):
    """Generate text response from the model."""
    with torch.no_grad():
        # Process image and question
        pixel_values = model.process_image(image_tensor)
        tokenized = model.tokenize(question)
        
        # Generate response
        generation_output = model.generate(
            pixel_values=pixel_values,
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=50
        )
        
        # Decode response
        response_ids = generation_output.sequences[0]
        response_text = model.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # Remove the input question from response
        if question in response_text:
            response_text = response_text.replace(question, "").strip()
    
    return response_text


def create_attention_visualization(original_image, attention_map, output_path):
    """Create and save attention map visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Attention map
    attention_resized = F.interpolate(
        attention_map.unsqueeze(0).unsqueeze(0), 
        size=original_image.size[::-1], 
        mode='bilinear', 
        align_corners=False
    ).squeeze().cpu().numpy()
    
    im1 = axes[1].imshow(attention_resized, cmap='hot', alpha=0.8)
    axes[1].set_title('Attention Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1])
    
    # Overlay
    axes[2].imshow(original_image)
    axes[2].imshow(attention_resized, cmap='hot', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention visualization saved to: {output_path}")


def save_attention_heatmap(attention_map, output_path):
    """Save just the attention heatmap as an image."""
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_map.cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.title('Attention Heatmap')
    plt.axis('off')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Attention heatmap saved to: {output_path}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load trained model
    model = load_trained_model(args.model_path)
    
    # Load and preprocess image
    original_image, processed_image, original_size = preprocess_input_image(
        args.image_path, args.image_size
    )
    
    print(f"Processing image: {args.image_path}")
    print(f"Original size: {original_size}")
    print(f"Processed size: {processed_image.shape}")
    
    # Extract attention map
    print("Extracting attention map...")
    attention_map = extract_attention_map(model, processed_image, args.question)
    
    # Generate text response if requested
    if args.generate_text:
        print("Generating text response...")
        text_response = generate_text_response(model, processed_image, args.question)
        print(f"Question: {args.question}")
        print(f"Response: {text_response}")
        
        # Save text response
        response_file = os.path.join(args.output_dir, "response.txt")
        with open(response_file, 'w') as f:
            f.write(f"Question: {args.question}\n")
            f.write(f"Response: {text_response}\n")
        print(f"Text response saved to: {response_file}")
    
    # Save attention visualization if requested
    if args.save_attention:
        print("Creating attention visualization...")
        
        # Full visualization with original image
        viz_path = os.path.join(args.output_dir, "attention_visualization.png")
        create_attention_visualization(original_image, attention_map, viz_path)
        
        # Just the heatmap
        heatmap_path = os.path.join(args.output_dir, "attention_heatmap.png")
        save_attention_heatmap(attention_map, heatmap_path)
    
    # Save attention map as numpy array
    attention_numpy = attention_map.cpu().numpy()
    np_path = os.path.join(args.output_dir, "attention_map.npy")
    np.save(np_path, attention_numpy)
    print(f"Attention map array saved to: {np_path}")
    
    print("\nInference completed successfully!")
    print(f"Results saved in: {args.output_dir}")


if __name__ == "__main__":
    main()