# InternVL2 Gaze Prediction Training

This repository contains code for training InternVL2 models for gaze prediction tasks using attention supervision.

## Files Overview

### Core Training Files
- `train.py` - Main training script
- `models/internvl_wrapper.py` - InternVL model wrapper with LoRA/QLoRA support
- `data/dataset.py` - Dataset classes for loading images and gaze maps
- `losses.py` - Loss functions for attention supervision (MSE, KL, Focal)
- `utils.py` - Utility functions for training

### Inference and Evaluation
- `inference.py` - Single image inference with attention visualization
- `evaluate.py` - Batch evaluation with metrics and visualizations

## Installation

```bash
# Install required packages
pip install torch torchvision transformers
pip install peft accelerate
pip install matplotlib numpy pillow scipy scikit-learn
pip install bitsandbytes  # For QLoRA support
```

## Usage

### Training

```bash
python train.py \
    --model_name_or_path "OpenGVLab/InternVL-8B" \
    --image_path "./data/images" \
    --gaze_path "./data/gaze_maps" \
    --output_dir "./output" \
    --loss_fn "kl" \
    --batch_size 4 \
    --lr 1e-5 \
    --epochs 5 \
    --image_size 224
```

**Arguments:**
- `--model_name_or_path`: HuggingFace model identifier or local path
- `--image_path`: Directory containing training images
- `--gaze_path`: Directory containing corresponding gaze maps
- `--output_dir`: Directory to save model checkpoints
- `--loss_fn`: Loss function (`kl`, `mse`, `focal`)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--epochs`: Number of training epochs
- `--image_size`: Size to resize images to

### Single Image Inference

```bash
python inference.py \
    --model_path "./output/checkpoint-5" \
    --image_path "./test_image.jpg" \
    --output_dir "./inference_results" \
    --save_attention \
    --generate_text \
    --question "Did the number of beds remain over 200,000 between 2010 and 2019?"
```

**Arguments:**
- `--model_path`: Path to trained model checkpoint
- `--image_path`: Path to input image
- `--output_dir`: Directory to save results
- `--save_attention`: Save attention visualization
- `--generate_text`: Generate text response
- `--question`: Question to ask about the image

**Outputs:**
- `attention_visualization.png`: Combined visualization with original image, attention map, and overlay
- `attention_heatmap.png`: Just the attention heatmap
- `attention_map.npy`: Raw attention map as numpy array
- `response.txt`: Generated text response (if --generate_text is used)

### Batch Evaluation

For simple gaze datasets:
```bash
python evaluate.py \
    --model_path "./output/checkpoint-5" \
    --test_image_path "./test_data/images" \
    --test_gaze_path "./test_data/gaze_maps" \
    --output_dir "./evaluation_results" \
    --dataset_type "simple" \
    --num_samples 10
```

For VQA datasets with Yes/No questions:
```bash
python evaluate.py \
    --model_path "./output/checkpoint-5" \
    --data_dir "./test_data" \
    --output_dir "./evaluation_results" \
    --dataset_type "vqa" \
    --num_samples 10
```

**Arguments:**
- `--dataset_type`: Type of evaluation (`simple` for gaze-only, `vqa` for questions+gaze)
- `--data_dir`: Directory with VQA data (contains images/, gaze_maps/, metadata.json)
- `--test_image_path`: Directory with test images (for simple mode)
- `--test_gaze_path`: Directory with test gaze maps (for simple mode)
- `--num_samples`: Number of samples to visualize

**Outputs:**
- `evaluation_results.txt`: Quantitative metrics summary
- `attention_comparison.png`: Visual comparison of predictions vs ground truth
- `metrics_distribution.png`: Distribution plots of evaluation metrics

## Data Format

### Directory Structure
```
data/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── gaze_maps/
    ├── image1.png
    ├── image2.png
    └── ...
```

### Requirements
- **Images**: RGB images in standard formats (JPG, PNG)
- **Gaze Maps**: Grayscale images (PNG recommended) with same base filename as corresponding image
- **Naming**: Gaze map should have same name as image but with .png extension

## Model Architecture

The model uses InternVL2 with:
- **LoRA/QLoRA**: For efficient fine-tuning
- **Cross-attention extraction**: Attention maps from vision-language interactions
- **Attention supervision**: Training attention maps to match human gaze patterns

## Loss Functions

1. **KL Divergence** (`kl`): Treats attention and gaze as probability distributions
2. **MSE** (`mse`): Direct pixel-wise regression
3. **Focal Loss** (`focal`): Handles attention distribution imbalance

## Evaluation Metrics

### Attention-based Metrics
1. **Pearson's Correlation Coefficient (CC)** - Linear correlation between attention and gaze maps
2. **Kullback–Leibler (KL) Divergence** - Distribution similarity measure 
3. **Histogram Intersection (SIM)** - Overlap between attention and gaze distributions

### Binary Classification Metrics (for Yes/No VQA)
4. **Accuracy** - Percentage of correct Yes/No answers

## Tips for Good Results

1. **Data Quality**: Ensure gaze maps are well-aligned with images
2. **Loss Function**: Start with KL divergence for attention-like tasks
3. **Learning Rate**: Use small learning rates (1e-5 to 1e-4) for fine-tuning
4. **Batch Size**: Adjust based on GPU memory (2-8 typical range)
5. **Image Size**: 224x224 is standard, larger sizes may improve quality but require more memory

## Troubleshooting

### Memory Issues
- Reduce batch size
- Use gradient accumulation
- Enable QLoRA (4-bit quantization)

### Poor Convergence
- Lower learning rate
- Try different loss functions
- Check data quality and alignment

### Import Errors
Make sure all required packages are installed:
```bash
pip install torch torchvision transformers peft accelerate matplotlib numpy pillow scipy scikit-learn bitsandbytes
```

## Example Workflow

```bash
python train.py \
    --image_path "./data/train/images" \
    --gaze_path "./data/train/gaze_maps" \
    --output_dir "./models/gaze_model" \
    --epochs 10

python inference.py \
    --model_path "./models/gaze_model/checkpoint-10" \
    --image_path "./test_image.jpg" \
    --save_attention \
    --generate_text

python evaluate.py \
    --model_path "./models/gaze_model/checkpoint-10" \
    --test_image_path "./data/test/images" \
    --test_gaze_path "./data/test/gaze_maps" \
    --dataset_type "simple"