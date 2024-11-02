import torch
import torch.nn as nn
from torch import Tensor


class OnlineLabelSmoothing(nn.Module):
    """
    Implements Online Label Smoothing from paper
    https://arxiv.org/pdf/2011.12562.pdf
    
    This module adapts label smoothing dynamically during training by learning
    from model predictions. It combines traditional cross entropy loss with
    a soft loss using learned label distributions.
    
    Args:
        alpha (float): Weight balancing factor between hard and soft loss. Default: 0.5
        n_classes (int): Number of classes in classification task. Default: 1000
        smoothing (float): Initial smoothing factor for first epoch. Default: 0.1
    """

    def __init__(self, alpha: float = 0.5, n_classes: int = 1000, smoothing: float = 0.1):
        super().__init__()
        assert 0 <= alpha <= 1, 'Alpha must be in range [0, 1]'
        self.a = alpha
        self.n_classes = n_classes
        
        # Initialize supervision matrix with standard label smoothing
        self.register_buffer('supervise', torch.full((n_classes, n_classes), 
                                                   smoothing / (n_classes - 1)))
        self.supervise.fill_diagonal_(1 - smoothing)

        # Buffers for updating supervision matrix
        self.register_buffer('update', torch.zeros_like(self.supervise))
        self.register_buffer('idx_count', torch.zeros(n_classes))
        
        self.hard_loss = nn.CrossEntropyLoss()

    def forward(self, y_h: Tensor, y: Tensor) -> Tensor:
        """
        Compute combined hard and soft loss.
        
        Args:
            y_h (Tensor): Predicted logits of shape (batch_size, n_classes)
            y (Tensor): Ground truth labels of shape (batch_size,)
            
        Returns:
            Tensor: Weighted sum of hard and soft loss
        """
        device = y_h.device
        self.to(device)
        
        soft_loss = self.soft_loss(y_h, y)
        hard_loss = self.hard_loss(y_h, y)
        return self.a * hard_loss + (1 - self.a) * soft_loss
    
    def _convert_one_hot_to_label(self, y_h: Tensor, y: Tensor) -> Tensor:
        """
        Convert one-hot encoded labels to class indices.
        
        Args:
            y_h (Tensor): Input tensor that may be one-hot encoded
            y (Tensor): Original label tensor
            
        Returns:
            Tensor: Class indices tensor
        """
        # Check if y_h is one-hot encoded by checking if each row sums to 1
        row_sums = y_h.sum(dim=-1)
        is_one_hot = torch.allclose(row_sums, torch.ones_like(row_sums))
        
        if is_one_hot:
            return y_h.argmax(dim=-1)
        return y

    def soft_loss(self, y_h: Tensor, y: Tensor) -> Tensor:
        """
        Calculate soft loss and update supervision matrix if training.
        
        Args:
            y_h (Tensor): Predicted logits of shape (batch_size, n_classes)
            y (Tensor): Ground truth labels of shape (batch_size,)
            
        Returns:
            Tensor: Soft loss value
        """
        y = self._convert_one_hot_to_label(y_h, y)
        y_h = y_h.log_softmax(dim=-1)
        if self.training:
            with torch.no_grad():
                self.step(y_h.exp(), y)
        true_dist = torch.index_select(self.supervise, 1, y).swapaxes(-1, -2)
        return torch.mean(torch.sum(-true_dist * y_h, dim=-1))

    def step(self, y_h: Tensor, y: Tensor) -> None:
        """
        Updates `update` with the probabilities
        of the correct predictions and updates `idx_count` counter for
        later normalization.

        Steps:
            1. Calculate correct classified examples.
            2. Filter `y_h` based on the correct classified.
            3. Add `y_h_f` rows to the `j` (based on y_h_idx) column of `memory`.
            4. Keep count of # samples added for each `y_h_idx` column.
            5. Average memory by dividing column-wise by result of step (4).

        Note on (5): This is done outside this function since we only need to
                     normalize **at the end of the epoch.**
        Args:
            y_h (Tensor): Predicted probabilities after softmax
            y (Tensor): Ground truth labels
        """
        y_h_idx = y_h.argmax(dim=-1)
        mask = torch.eq(y_h_idx, y)
        y_h_c = y_h[mask]
        y_h_idx_c = y_h_idx[mask]
        
        self.update.index_add_(1, y_h_idx_c, y_h_c.swapaxes(-1, -2))
        self.idx_count.index_add_(0, y_h_idx_c, 
                                 torch.ones_like(y_h_idx_c, dtype=torch.float32))

    def next_epoch(self) -> None:
        """
        Update supervision matrix for next epoch.
        Should be called at the end of each epoch.
        """
        # Avoid division by zero
        self.idx_count[torch.eq(self.idx_count, 0)] = 1
        
        # Normalize update matrix and transfer to supervision matrix
        self.update /= self.idx_count
        self.supervise = self.update
        
        # Reset update matrix and counts
        self.update = self.update.clone().zero_()
        self.idx_count.zero_()