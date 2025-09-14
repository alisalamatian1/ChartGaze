import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_map, gaze_map):
        attn_map = attn_map.view(attn_map.size(0), -1)
        gaze_map = gaze_map.view(gaze_map.size(0), -1)
        loss = F.mse_loss(attn_map, gaze_map, reduction='mean')
        return loss

class AttentionKLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_map, gaze_map):
        attn_map = attn_map.view(attn_map.size(0), -1)
        gaze_map = gaze_map.view(gaze_map.size(0), -1)
        attn_map = F.softmax(attn_map, dim=-1)
        gaze_map = F.softmax(gaze_map, dim=-1)
        loss = F.kl_div(attn_map.log(), gaze_map, reduction='batchmean')
        return loss

class AttentionFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, attn_map, gaze_map):
        attn_map = attn_map.view(attn_map.size(0), -1)
        gaze_map = gaze_map.view(gaze_map.size(0), -1)
        attn_map = torch.clamp(attn_map, min=self.eps, max=1.0 - self.eps)
        loss = -self.alpha * (1 - attn_map) ** self.gamma * gaze_map * attn_map.log()
        return loss.mean()

def get_loss_fn(name: str):
    if name == "mse":
        return AttentionMSELoss()
    elif name == "kl":
        return AttentionKLLoss()
    elif name == "focal":
        return AttentionFocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
