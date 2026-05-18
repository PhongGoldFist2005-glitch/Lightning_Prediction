import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_pred, y):
        y_pred = y_pred.squeeze()
        y = y.squeeze()
        
        bce = nn.functional.binary_cross_entropy_with_logits(
            y_pred, y.float(), reduction='none'
        )
        pt = torch.exp(-bce)
        alpha_t = self.alpha * y.float() + (1 - self.alpha) * (1 - y.float())
        focal = alpha_t * ((1 - pt) ** self.gamma) * bce
        
        return focal.mean()