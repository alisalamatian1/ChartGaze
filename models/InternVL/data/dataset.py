import os
import json
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

class GazeDataset(Dataset):
    """Simple dataset for gaze prediction training."""
    def __init__(self, image_path, gaze_path, image_size=224):
        self.image_path = image_path
        self.gaze_path = gaze_path
        self.image_size = image_size
        
        # Get list of image files
        self.image_files = []
        if os.path.isdir(image_path):
            for f in os.listdir(image_path):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(f)
        else:
            # Single file
            self.image_files = [os.path.basename(image_path)]
            self.image_path = os.path.dirname(image_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_path, img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Load corresponding gaze map
        gaze_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        gaze_map_path = os.path.join(self.gaze_path, gaze_name)
        
        if os.path.exists(gaze_map_path):
            gaze_map = Image.open(gaze_map_path).convert('L')
            gaze_map = transforms.Resize((self.image_size, self.image_size))(gaze_map)
            gaze_map = transforms.ToTensor()(gaze_map)
        else:
            # Create dummy gaze map if not found
            gaze_map = torch.zeros(1, self.image_size, self.image_size)
        
        return image, gaze_map

class InternVLGazeDataset(Dataset):
    def __init__(self, data_dir, processor, max_length=512):
        self.data = []
        self.processor = processor
        self.max_length = max_length
        image_dir = os.path.join(data_dir, "images")
        gaze_dir = os.path.join(data_dir, "gaze_maps")
        meta_path = os.path.join(data_dir, "metadata.json")

        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        for item in metadata:
            image_path = os.path.join(image_dir, item["image"])
            gaze_path = os.path.join(gaze_dir, item["gaze"])
            if os.path.exists(image_path) and os.path.exists(gaze_path):
                self.data.append({
                    "image_path": image_path,
                    "gaze_path": gaze_path,
                    "question": item["question"],
                    "answer": item["answer"]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        gaze_map = Image.open(item["gaze_path"]).convert("L")
        processor_outputs = self.processor(images=image, text=item["question"], return_tensors="pt", padding="max_length", max_length=self.max_length, truncation=True)
        pixel_values = processor_outputs["pixel_values"].squeeze(0)
        input_ids = processor_outputs["input_ids"].squeeze(0)
        attention_mask = processor_outputs["attention_mask"].squeeze(0)
        gaze_tensor = torch.tensor(gaze_map, dtype=torch.float32).unsqueeze(0) / 255.0
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "gaze_map": gaze_tensor,
            "answer": item["answer"]
        }
