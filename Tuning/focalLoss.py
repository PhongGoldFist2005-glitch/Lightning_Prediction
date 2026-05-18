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

class FocalLoss6Output(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, y_pred, y):
        # y_pred = y_pred.squeeze()
        # y = y.squeeze()
        
        bce = nn.functional.binary_cross_entropy_with_logits(
            y_pred, y.float(), reduction='none'
        )
        
        pt = torch.exp(-bce)
        alpha_t = self.alpha * y.float() + (1 - self.alpha) * (1 - y.float())
        focal = alpha_t * ((1 - pt) ** self.gamma) * bce
        
        return focal.mean()

class FocalLossMultiTask(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, init_log_var_reg=-3.0, init_log_var_cls=-3.0, device='cpu'):
        """Multi-task loss with learnable log-variances.

        Args:
            alpha: focal alpha
            gamma: focal gamma
            init_log_var_reg: initial value for log variance of regression term
            init_log_var_cls: initial value for log variance of classification term
            device: device hint (module will be moved with `.to(device)`)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

        # Learnable log-variance parameters. Initialize away from 0.0 to help
        # early-stage learning (default -3 -> moderate initial weighting).
        self.log_var_reg = nn.Parameter(torch.tensor([init_log_var_reg], dtype=torch.float32)).to(device)
        self.log_var_cls = nn.Parameter(torch.tensor([init_log_var_cls], dtype=torch.float32)).to(device)
    
    def focal_loss(self, y_pred_cls, y_cls):
        y_cls_f = y_cls.float().squeeze()
        y_pred_cls = y_pred_cls.squeeze()
        
        bce = F.binary_cross_entropy_with_logits(
            y_pred_cls, y_cls_f, reduction='none'
        )
        pt = torch.exp(-bce)
        alpha_t = self.alpha * y_cls_f + (1 - self.alpha) * (1 - y_cls_f)
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()
    
    def forward(self, y_pred_reg, y_pred_cls, y_reg, y_cls):
        y_pred_reg = y_pred_reg.squeeze(-1)
        
        # Individual losses
        mse_loss = F.mse_loss(y_pred_reg, y_reg.float(), reduction='mean')
        f_loss   = self.focal_loss(y_pred_cls, y_cls)
        
        # ✅ Uncertainty weighting — model tự học cân bằng
        precision_reg = torch.exp(-self.log_var_reg)
        precision_cls = torch.exp(-self.log_var_cls)
        
        total_loss = (
            precision_reg * mse_loss + self.log_var_reg * 0.5 +
            precision_cls * f_loss   + self.log_var_cls * 0.5
        )
        
        return {
            'loss':       total_loss,
            'mse_loss':   mse_loss,
            'focal_loss': f_loss,
            # Monitor xem model đang weight như thế nào
            'w_reg': precision_reg.item(),
            'w_cls': precision_cls.item(),
        }