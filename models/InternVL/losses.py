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


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE Loss for gaze supervision.
    
    The weight formula is: w_i = 1 / (alpha - G_i)
    where G_i is the ground truth gaze value (normalized to [0, 1]).
    
    This gives higher weight to regions with higher gaze values,
    making the model focus more on salient regions.
    
    Loss = sum(w_i * (G_i - A_i)^2)
    
    Both gaze_map and attn_map should be normalized to [0, 1] before calling this loss.
    """
    def __init__(self, alpha=1.1):
        super().__init__()
        self.alpha = alpha  # Must be > 1 to avoid division by zero when gaze=1

    def forward(self, attn_map, gaze_map):
        """
        Args:
            attn_map: Model's attention map, should be normalized to [0, 1]
            gaze_map: Ground truth gaze heatmap, should be normalized to [0, 1]
        """
        attn_map = attn_map.view(attn_map.size(0), -1)
        gaze_map = gaze_map.view(gaze_map.size(0), -1)
        
        # Compute weights: higher gaze values get higher weights
        # w = 1 / (alpha - gaze), where alpha > 1
        w = 1.0 / (self.alpha - gaze_map)
        
        # Weighted MSE
        w_mse = torch.sum(w * (gaze_map - attn_map) ** 2)
        return w_mse


def normalize_attention_map(attn_map, skip_top_k=0):
    """
    Normalize attention map to [0, 1] range.
    
    Args:
        attn_map: Attention map tensor of shape (batch_size, seq_len) or (batch_size, H, W)
        skip_top_k: Number of top-K values to zero out before normalization (reduces outlier impact)
    
    Returns:
        Normalized attention map in [0, 1] range
    """
    # Flatten if needed
    original_shape = attn_map.shape
    if len(attn_map.shape) > 2:
        attn_map = attn_map.view(attn_map.size(0), -1)
    
    # Optionally remove top-K outliers
    if skip_top_k > 0:
        _, topk_indices = torch.topk(attn_map, skip_top_k, dim=-1)
        attn_map = attn_map.scatter(dim=-1, index=topk_indices, value=0)
    
    # Max normalization per sample
    max_vals = attn_map.max(dim=-1, keepdim=True)[0]
    max_vals = torch.clamp(max_vals, min=1e-8)  # Avoid division by zero
    attn_map = attn_map / max_vals
    
    return attn_map.view(original_shape)


def normalize_gaze_map(gaze_map):
    """
    Normalize gaze heatmap to [0, 1] range using max normalization.
    
    Args:
        gaze_map: Gaze heatmap tensor
    
    Returns:
        Normalized gaze map in [0, 1] range
    """
    # Flatten for normalization
    original_shape = gaze_map.shape
    gaze_flat = gaze_map.view(gaze_map.size(0), -1)
    
    # Max normalization
    max_vals = gaze_flat.max(dim=-1, keepdim=True)[0]
    max_vals = torch.clamp(max_vals, min=1e-8)  # Avoid division by zero
    gaze_normalized = gaze_flat / max_vals
    
    return gaze_normalized.view(original_shape)

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

def get_loss_fn(name: str, **kwargs):
    if name == "mse":
        return AttentionMSELoss()
    elif name == "wmse" or name == "weighted_mse":
        alpha = kwargs.get('alpha', 1.1)
        return WeightedMSELoss(alpha=alpha)
    elif name == "kl":
        return AttentionKLLoss()
    elif name == "focal":
        return AttentionFocalLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
