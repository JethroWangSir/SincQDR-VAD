import math
from torch.optim.lr_scheduler import _LRScheduler
import os
import logging
import torch
import torch.nn as nn
from scipy.ndimage import median_filter
from sklearn.metrics import confusion_matrix, roc_auc_score, fbeta_score
import numpy as np
from torchinfo import summary
from torchprofile import profile_macs
from typing import Dict, Tuple
from collections import OrderedDict
from thop import profile
from tqdm import tqdm

class WarmupHoldDecayScheduler(_LRScheduler):
    """
    Implements WarmupHold-Decay learning rate scheduler with the following phases:
    1. Linear warmup
    2. Constant hold
    3. Polynomial decay
    
    Args:
        optimizer: Wrapped optimizer
        total_steps: Total number of training steps
        warmup_ratio: Percentage of steps for warmup phase (default: 0.05)
        hold_ratio: Percentage of steps for hold phase (default: 0.45)
        min_lr: Minimum learning rate (default: 0.001)
        max_lr: Maximum learning rate (default: 0.01)
        power: Power for polynomial decay (default: 2.0)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, total_steps, warmup_ratio=0.05, hold_ratio=0.45,
                 min_lr=0.001, max_lr=0.01, power=2.0, last_epoch=-1):
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.hold_steps = int(total_steps * hold_ratio)
        self.decay_steps = total_steps - self.warmup_steps - self.hold_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)

        step = self.last_epoch

        # Warmup phase
        if step < self.warmup_steps:
            lr_scale = step / self.warmup_steps
            return [self.min_lr + (self.max_lr - self.min_lr) * lr_scale for _ in self.base_lrs]
        
        # Hold phase
        elif step < self.warmup_steps + self.hold_steps:
            return [self.max_lr for _ in self.base_lrs]
        
        # Decay phase
        else:
            decay_steps_completed = step - self.warmup_steps - self.hold_steps
            decay_ratio = decay_steps_completed / self.decay_steps
            decay_factor = (1 - decay_ratio) ** self.power
            
            lr = self.min_lr + (self.max_lr - self.min_lr) * decay_factor
            return [lr for _ in self.base_lrs]


def save_best_k_model_with_auroc(exp_dir, model, epoch, auroc, top_k_auc_scores, k=3):
    checkpoint_path = os.path.join(exp_dir, f'model_epoch_{epoch + 1}_val_auroc={auroc:.4f}.ckpt')
    torch.save(model.state_dict(), checkpoint_path)
    logging.info(f'Model saved to {checkpoint_path}')

    top_k_auc_scores.append((auroc, checkpoint_path))
    top_k_auc_scores.sort(key=lambda x: x[0], reverse=True)

    if len(top_k_auc_scores) > k:
        _, worst_model_path = top_k_auc_scores.pop()
        if os.path.exists(worst_model_path):
            os.remove(worst_model_path)
            logging.info(f'Removed model: {worst_model_path}')


def model_info(sinc_conv, model, num_samples, frame_size, device):
    if sinc_conv:
        params_count = summary(model, input_size=(1, 1, num_samples), verbose=0).total_params / 1_000
        macs = profile_macs(model, torch.randn(1, 1, num_samples).to(device)) / 1_000_000
    else:
        params_count = summary(model, input_size=(1, 1, 64, frame_size), verbose=0).total_params / 1_000
        macs = profile_macs(model, torch.randn(1, 1, 64, frame_size).to(device)) / 1_000_000
    
    return params_count, macs


def median_smoothing_filter(y_pred, y_true, y_pred_list, y_true_list, median_kernel_size, device):
    y_pred = y_pred.cpu().numpy()
    y_pred_smoothed = torch.tensor(median_filter(y_pred, size=(median_kernel_size, 1))).to(device)
    y_pred_list.append(y_pred_smoothed)
    y_true_list.append(y_true)

    return y_pred_list, y_true_list


def calculate_fpr_fnr(y_true, y_pred, threshold):
    # Apply threshold to predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Calculate FPR and FNR
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    return fpr, fnr


def metrics_calculation(y_true, y_pred, threshold):
    auroc = roc_auc_score(y_true, y_pred)
    fpr, fnr = calculate_fpr_fnr(y_true, y_pred, threshold)
    y_pred_binary = torch.tensor(y_pred >= threshold).float().cpu().numpy()
    f2_score = fbeta_score(y_true, y_pred_binary, beta=2)

    # print(f'val_labels_cat: {y_true.shape}, val_probs_cat: {y_pred.shape}, binarized_preds: {y_pred_binary.shape}')

    return auroc, fpr, fnr, f2_score
