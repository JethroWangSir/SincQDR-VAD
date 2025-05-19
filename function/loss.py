import torch
import torch.nn as nn
import numpy as np

class QDRLoss(nn.Module):
    """
    QDRLoss class to compute either Pairwise Square Loss (PSQ) or Composite Square Loss (CSQ).
    
    Args:
        loss_type (str): Type of loss to compute ('psq' or 'csq').
        margin (float): Hyperparameter c in the loss function.
    """
    def __init__(self, loss_type='psq', margin=1.0):
        super(QDRLoss, self).__init__()
        assert loss_type in ['psq', 'csq'], "loss_type must be 'psq' or 'csq'"
        self.loss_type = loss_type
        self.margin = margin

    def forward(self, preds, labels):
        """
        Forward method to compute the loss.
        
        Args:
            preds (Tensor): Model predictions (1D or 2D tensor).
            labels (Tensor): Ground-truth labels (1D or 2D tensor, 0 or 1).
        
        Returns:
            loss (Tensor): The computed loss value.
        """
        # Ensure predictions and labels are 1D
        preds = preds.flatten()
        labels = labels.flatten()

        # Split positive and negative samples
        pos_mask = labels == 1
        neg_mask = labels == 0

        pos_preds = preds[pos_mask]
        neg_preds = preds[neg_mask]

        if len(pos_preds) == 0 or len(neg_preds) == 0:
            # If no positive or negative samples, return zero loss
            return torch.tensor(0.0, device=preds.device)

        if self.loss_type == 'psq':
            # Pairwise Square Loss (PSQ)
            pairwise_diffs = pos_preds.unsqueeze(1) - neg_preds.unsqueeze(0)
            loss = torch.clamp(self.margin - pairwise_diffs, min=0.0) ** 2
            return loss.mean()

        elif self.loss_type == 'csq':
            # Composite Square Loss (CSQ)
            a = pos_preds.mean()  # Mean prediction for positive samples
            b = neg_preds.mean()  # Mean prediction for negative samples

            term1 = ((pos_preds - a) ** 2).mean()  # Variance for positive samples
            term2 = ((neg_preds - b) ** 2).mean()  # Variance for negative samples
            term3 = 0.5 * (self.margin + b - a) ** 2  # Squared difference term

            return term1 + term2 + term3
