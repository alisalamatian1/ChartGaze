import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import pearsonr
import torch.nn.functional as F

from models.internvl_wrapper import load_internvl_model
from data.dataset import GazeDataset, InternVLGazeDataset
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--test_image_path', type=str, required=True,
                       help='Path to test images directory')
    parser.add_argument('--test_gaze_path', type=str, required=True,
                       help='Path to test gaze maps directory')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--image_size', type=int, default=224,
                       help='Size to resize images')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--dataset_type', type=str, choices=['simple', 'vqa'], default='simple',
                       help='Type of dataset: simple (image+gaze) or vqa (image+gaze+questions)')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Data directory for VQA dataset (contains images/, gaze_maps/, metadata.json)')
    return parser.parse_args()


def load_trained_model(model_path, base_model_name="OpenGVLab/InternVL-8B"):
    """Load the trained model from checkpoint."""
    print(f"Loading model from {model_path}")
    
    model = load_internvl_model(base_model_name, qlora=True)
    
    if os.path.isdir(model_path):
        from peft import PeftModel
        model.model = PeftModel.from_pretrained(model.model, model_path)
    else:
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
    
    model.eval()
    return model


def calculate_metrics(pred_attention, true_gaze, pred_answer=None, true_answer=None):
    """Calculate various metrics between predicted attention and true gaze."""
    # Flatten arrays for attention-based metrics
    pred_flat = pred_attention.flatten()
    true_flat = true_gaze.flatten()
    
    # Normalize maps for probability-based metrics
    pred_norm = pred_flat / (pred_flat.sum() + 1e-8)
    true_norm = true_flat / (true_flat.sum() + 1e-8)
    
    # 1. Pearson's Correlation Coefficient (CC)
    correlation, _ = pearsonr(pred_flat, true_flat)
    if np.isnan(correlation):
        correlation = 0.0
    
    # 2. Kullback–Leibler (KL) divergence
    # KL(P||Q) where P is true distribution, Q is predicted distribution
    kl_divergence = np.sum(true_norm * np.log((true_norm + 1e-8) / (pred_norm + 1e-8)))
    
    # 3. Histogram Intersection (Similarity/SIM metric)
    # Also known as Bhattacharyya coefficient for normalized histograms
    histogram_intersection = np.sum(np.minimum(pred_norm, true_norm))
    
    metrics = {
        'correlation': correlation,
        'kl_divergence': kl_divergence,
        'histogram_intersection': histogram_intersection,
    }
    
    # 4. Accuracy for Yes/No binary classification (if answers are provided)
    if pred_answer is not None and true_answer is not None:
        # Convert to binary predictions if needed
        if isinstance(pred_answer, (float, np.floating)):
            pred_binary = 1 if pred_answer > 0.5 else 0
        else:
            pred_binary = pred_answer
            
        if isinstance(true_answer, (float, np.floating)):
            true_binary = 1 if true_answer > 0.5 else 0
        else:
            true_binary = true_answer
            
        accuracy = 1.0 if pred_binary == true_binary else 0.0
        metrics['accuracy'] = accuracy
    
    return metrics


def visualize_comparison(images, pred_attentions, true_gazes, metrics_list, output_dir, num_samples=5):
    """Create comparison visualizations."""
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Original image
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # True gaze
        true_gaze = true_gazes[i].squeeze().cpu().numpy()
        im1 = axes[i, 1].imshow(true_gaze, cmap='hot')
        axes[i, 1].set_title('Ground Truth Gaze')
        axes[i, 1].axis('off')
        
        # Predicted attention
        pred_attn = pred_attentions[i].squeeze().cpu().numpy()
        im2 = axes[i, 2].imshow(pred_attn, cmap='hot')
        axes[i, 2].set_title('Predicted Attention')
        axes[i, 2].axis('off')
        
        # Difference map
        diff = np.abs(true_gaze - pred_attn)
        im3 = axes[i, 3].imshow(diff, cmap='viridis')
        metrics_text = f'CC: {metrics_list[i]["correlation"]:.3f}\nKL: {metrics_list[i]["kl_divergence"]:.3f}\nSIM: {metrics_list[i]["histogram_intersection"]:.3f}'
        if 'accuracy' in metrics_list[i]:
            metrics_text += f'\nAcc: {metrics_list[i]["accuracy"]:.3f}'
        axes[i, 3].set_title(f'Difference\n{metrics_text}')
        axes[i, 3].axis('off')
        
        # Add colorbars
        plt.colorbar(im1, ax=axes[i, 1], shrink=0.6)
        plt.colorbar(im2, ax=axes[i, 2], shrink=0.6)
        plt.colorbar(im3, ax=axes[i, 3], shrink=0.6)
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'attention_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison visualization saved to: {comparison_path}")


def plot_metrics_distribution(all_metrics, output_dir):
    """Plot distribution of metrics across all samples."""
    # Check which metrics are available
    available_metrics = set(all_metrics[0].keys())
    
    if 'accuracy' in available_metrics:
        # Plot 2x2 grid with accuracy
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        metrics = ['correlation', 'kl_divergence', 'histogram_intersection', 'accuracy']
        titles = ['Correlation (CC) Distribution', 'KL Divergence Distribution', 
                  'Histogram Intersection (SIM) Distribution', 'Accuracy Distribution']
    else:
        # Plot 1x3 grid without accuracy
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes = axes.flatten()
        metrics = ['correlation', 'kl_divergence', 'histogram_intersection']
        titles = ['Correlation (CC) Distribution', 'KL Divergence Distribution', 
                  'Histogram Intersection (SIM) Distribution']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        if len(axes.shape) == 1:
            ax = axes[i]
        else:
            ax = axes[i // 2, i % 2]
            
        values = [m[metric] for m in all_metrics if metric in m and not np.isnan(m[metric])]
        
        if values:
            ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{title}\nMean: {np.mean(values):.4f}, Std: {np.std(values):.4f}')
            ax.set_xlabel(metric.replace('_', ' ').title())
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No data for {metric}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
    
    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'metrics_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics distribution plot saved to: {dist_path}")


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_trained_model(args.model_path)
    device = next(model.parameters()).device
    
    # Create test dataset based on type
    if args.dataset_type == 'vqa':
        if args.data_dir is None:
            raise ValueError("--data_dir must be provided for VQA dataset type")
        test_dataset = InternVLGazeDataset(args.data_dir, model.processor, max_length=512)
        use_vqa = True
    else:
        test_dataset = GazeDataset(args.test_image_path, args.test_gaze_path, args.image_size)
        use_vqa = False
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    print(f"Dataset type: {args.dataset_type}")
    
    all_metrics = []
    sample_images = []
    sample_pred_attentions = []
    sample_true_gazes = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if use_vqa:
                # VQA dataset returns dict with pixel_values, gaze_map, answer, etc.
                images = batch["pixel_values"].to(device)
                true_gazes = batch["gaze_map"].to(device)
                answers = batch["answer"] if "answer" in batch else None
            else:
                # Simple dataset returns tuple (images, gazes)
                images, true_gazes = batch
                images = images.to(device)
                true_gazes = true_gazes.to(device)
                answers = None
            
            # Get attention predictions
            outputs = model.forward_with_attention(images, return_attn=True)
            pred_attentions = outputs['attn']
            
            # Get text predictions for VQA if available
            pred_answers = None
            if use_vqa and answers is not None:
                try:
                    # Generate yes/no answers
                    pred_answers = []
                    for i in range(images.size(0)):
                        # Use a simple yes/no question
                        question = "Is this a yes or no question? Answer yes or no."
                        response = generate_text_response_single(model, images[i], question)
                        # Convert response to binary (1 for yes, 0 for no)
                        pred_binary = 1 if 'yes' in response.lower() else 0
                        pred_answers.append(pred_binary)
                except Exception as e:
                    print(f"Warning: Could not generate text responses: {e}")
                    pred_answers = None
            
            # Resize predictions to match true gaze size
            if pred_attentions.shape[-2:] != true_gazes.shape[-2:]:
                pred_attentions = F.interpolate(
                    pred_attentions, 
                    size=true_gazes.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
            
            # Calculate metrics for each sample in batch
            for i in range(images.size(0)):
                pred_attn = pred_attentions[i].cpu().numpy()
                true_gaze = true_gazes[i].cpu().numpy()
                
                # Get answers for this sample
                true_answer = None
                pred_answer = None
                if answers is not None and pred_answers is not None:
                    true_answer = 1 if answers[i].lower() == 'yes' else 0
                    pred_answer = pred_answers[i]
                
                metrics = calculate_metrics(pred_attn, true_gaze, pred_answer, true_answer)
                all_metrics.append(metrics)
                
                # Store samples for visualization
                if len(sample_images) < args.num_samples:
                    sample_images.append(images[i].cpu())
                    sample_pred_attentions.append(pred_attentions[i].cpu())
                    sample_true_gazes.append(true_gazes[i].cpu())
            
            if batch_idx % 10 == 0:
                print(f"Processed {batch_idx * args.batch_size} samples...")
    
    # Calculate overall statistics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        if values:
            avg_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric, stats in avg_metrics.items():
        print(f"{metric.upper().replace('_', ' ')}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print()
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset type: {args.dataset_type}\n")
        f.write(f"Number of samples: {len(all_metrics)}\n\n")
        for metric, stats in avg_metrics.items():
            f.write(f"{metric.upper().replace('_', ' ')}:\n")
            f.write(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]\n")
            f.write("\n")
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_comparison(
        sample_images, sample_pred_attentions, sample_true_gazes, 
        all_metrics[:args.num_samples], args.output_dir, args.num_samples
    )
    
    plot_metrics_distribution(all_metrics, args.output_dir)
    
    print(f"\nEvaluation completed! Results saved in: {args.output_dir}")


def generate_text_response_single(model, image_tensor, question):
    """Generate text response for a single image."""
    try:
        with torch.no_grad():
            # Process image and question
            pixel_values = image_tensor.unsqueeze(0)
            tokenized = model.tokenize(question)
            
            # Generate response
            generation_output = model.generate(
                pixel_values=pixel_values,
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                max_new_tokens=10  # Short for yes/no answers
            )
            
            # Decode response
            response_ids = generation_output.sequences[0]
            response_text = model.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Remove the input question from response
            if question in response_text:
                response_text = response_text.replace(question, "").strip()
            
            return response_text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "no"  # Default to no


if __name__ == "__main__":
    main()