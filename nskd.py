import torch
import torch.nn as nn
import torch.nn.functional as F



#TODO : should support more models
#TODO : Add Choice for more layers 

class ViTFeatureExtractor(nn.Module):
    """A wrapper for ViT models that can be safely wrapped and unwrapped during DDP training.
    
    This implementation allows for:
    1. Safe feature extraction during training
    2. Easy wrapping/unwrapping without losing hooks
    3. Proper cleanup to prevent memory leaks
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = None
        self.hook = None
        self.register_hooks()

    def register_hooks(self):
        def hook(module, input, output):
            self.features = output[:, 0]
        
        # Store hook handle for later removal
        target_layer = self.model.module.blocks[int(0.5*len(self.model.module.blocks)) - 1] \
                      if hasattr(self.model, 'module') else \
                      self.model.blocks[int(0.5*len(self.model.module.blocks)) - 1]
        self.hook = target_layer.register_forward_hook(hook)

    def unwrap(self):
        """Safely unwrap the model and clean up hooks."""
        if self.hook is not None:
            self.hook.remove()
        return self.model

    @staticmethod
    def wrap(model):
        """Safely wrap a model, handling both DDP and non-DDP cases."""
        if hasattr(model, '') and hasattr(model, 'hook'):
            # check wrapped
            return model
        return ViTFeatureExtractor(model)

    def forward(self, x):
        outputs = self.model(x)
        return outputs, self.features
    
class USKDLoss(nn.Module):
    """
    
    This implements the USKD loss function which combines target supervision loss,
    non-target supervision loss and weak supervision loss.
    
    Args:
        channel (int): Feature dimension from the backbone model
        alpha (float): Weight for target supervision loss. Default: 1.0
        beta (float): Weight for non-target supervision loss. Default: 0.1 
        mu (float): Weight for weak supervision loss. Default: 0.005
        num_classes (int): Number of classes. Default: 1000
    """

    def __init__(self,
                 channel: int,
                 alpha: float = 1.0,
                 beta: float = 0.1,
                 mu: float = 0.005,
                 num_classes: int = 1000):
        super().__init__()

        self.channel = channel
        self.alpha = alpha
        self.beta = beta
        self.mu = mu
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(channel, num_classes)
        
        # Initialize Zipf distribution
        self.register_buffer('zipf', 1 / torch.arange(1, num_classes + 1))

    def forward(self, fea_mid: torch.Tensor, logit_s: torch.Tensor, 
                gt_label: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute USKD loss.
        
        Args:
            fea_mid: Mid-level features from backbone
            logit_s: Student model logits
            gt_label: Ground truth labels (one-hot or indices)
            
        Returns:
            Combined USKD loss
        """
        # Handle both one-hot and index labels
        if len(gt_label.size()) > 1:
            value, label = torch.sort(gt_label, descending=True, dim=-1)
            value = value[:,:2]
            label = label[:,:2]
        else:
            label = gt_label.view(len(gt_label), 1)
            value = torch.ones_like(label)

        N, c = logit_s.shape

        # Target supervision loss
        s_i = F.softmax(logit_s, dim=1)
        s_t = torch.gather(s_i, 1, label)

        p_t = s_t**2
        p_t = p_t + value - p_t.mean(0, True)
        p_t[value==0] = 0
        p_t = p_t.detach()

        s_i = self.log_softmax(logit_s)
        s_t = torch.gather(s_i, 1, label)
        loss_t = -(p_t * s_t).sum(dim=1).mean()

        # Weak supervision loss
        if len(gt_label.size()) > 1:
            target = gt_label * 0.9 + 0.1 * torch.ones_like(logit_s) / c
        else:
            target = torch.zeros_like(logit_s).scatter_(1, label, 0.9) + 0.1 / c
        
        w_fc = self.fc(fea_mid)
        w_i = self.log_softmax(w_fc)
        loss_weak = -(self.mu * target * w_i).sum(dim=1).mean()

        # Non-target supervision loss
        w_i = F.softmax(w_fc, dim=1)
        w_t = torch.gather(w_i, 1, label)

        # Compute rank scores
        rank = w_i / (1 - w_t.sum(1, True) + 1e-6) + s_i / (1 - s_t.sum(1, True) + 1e-6)
        _, rank = torch.sort(rank, descending=True, dim=-1)
        
        # Apply Zipf distribution
        z_i = self.zipf.repeat(N, 1)
        ids_sort = torch.argsort(rank)
        z_i = torch.gather(z_i, dim=1, index=ids_sort)

        # Mask out target classes
        mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
        logit_s = logit_s[mask].reshape(N, -1)
        ns_i = self.log_softmax(logit_s)

        nz_i = z_i[mask].reshape(N, -1)
        nz_i = nz_i / nz_i.sum(dim=1, keepdim=True)
        nz_i = nz_i.detach()
        
        loss_non = -(nz_i * ns_i).sum(dim=1).mean()

        # Combine losses
        loss_uskd = self.alpha * loss_t + self.beta * loss_non + loss_weak

        return loss_uskd