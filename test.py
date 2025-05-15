import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
from dataset import AVA_Tuple
from model.sincqdrvad import SincQDRVAD
from function.util import model_info, median_smoothing_filter, metrics_calculation

THRESHOLD = 0.5
WINDOW_SIZE = 0.63
SINC_CONV = True
QDR_LOSS_WEIGHT = 0.25
QDR_LOSS_TYPE = 'psq'

if WINDOW_SIZE == 0.63:
    overlap = 0.875
    patch_size = 8
    median_kernel_size = 7
    num_samples = 10080
    frame_size = 64
elif WINDOW_SIZE == 0.16:
    overlap = 0.8
    patch_size = 2
    median_kernel_size = 11
    num_samples = 2520
    frame_size = 16
elif WINDOW_SIZE == 0.032:
    overlap = 0.0
    patch_size = 1
    median_kernel_size = 21
    num_samples = 504
    frame_size = 4

# Set GPU and paths
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if SINC_CONV and QDR_LOSS_WEIGHT > 0.0:
    name = f'exp_{WINDOW_SIZE}_sinc_tinyvad_{QDR_LOSS_TYPE}_{QDR_LOSS_WEIGHT}'
    max_duration = 300.0
elif SINC_CONV:
    name = f'exp_{WINDOW_SIZE}_sinc_tinyvad'
    max_duration = 300.0
elif QDR_LOSS_WEIGHT > 0.0:
    name = f'exp_{WINDOW_SIZE}_tinyvad_{QDR_LOSS_TYPE}_{QDR_LOSS_WEIGHT}'
    max_duration = 300.0
else:
    name = f'exp_{WINDOW_SIZE}_tinyvad'
    max_duration = 300.0

exp_dir = f'./exp/{name}/'
os.makedirs(exp_dir, exist_ok=True)

# Initialize model and load the best checkpoint
model = SincVAD(1, 32, 64, patch_size, num_blocks, SINC_CONV).to(device)
checkpoint_path = os.path.join(exp_dir, 'model_last_epoch.ckpt')
model.load_state_dict(torch.load(checkpoint_path))
model.eval()


# Model information
params_count, macs = model_info(SINC_CONV, model, num_samples, frame_size, device)

model_info = {"Param Count (k)": params_count, "MACs (M)": macs}
with open(os.path.join(exp_dir, 'model_info.json'), 'w') as f:
    json.dump(model_info, f, indent=4)

print(f"Param Count: {params_count:.2f}k, MACs: {macs:.2f}M")


# Test AVA
print('Loading AVA test set ...')
test_dataset = AVA_Tuple(
    '/share/nas165/aaronelyu/Datasets/AVA-speech/',
    max_duration=max_duration,
    sample_duration=WINDOW_SIZE,
    overlap=overlap,
    feature_extraction=(not SINC_CONV),
)
print(f"Test dataset size: {len(test_dataset)}")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

val_probs_list, val_labels_list = [], []

# Test loop
with torch.no_grad():
    for batch in tqdm(test_loader, desc=f'Testing {name}'):
        if SINC_CONV:
            val_inputs = [item[0].to(device) for item in batch]
        else:
            val_inputs = [item[1].to(device) for item in batch]
        val_labels = [item[2].to(device).float().unsqueeze(1) for item in batch]
        val_inputs = torch.cat(val_inputs, dim=0)
        val_labels = torch.cat(val_labels, dim=0)

        val_probs = model.predict(val_inputs)

        # Apply median smoothing filter
        val_probs_list, val_labels_list = median_smoothing_filter(val_probs, val_labels, val_probs_list, val_labels_list, median_kernel_size, device)

# Concatenate results
val_labels_cat = torch.cat(val_labels_list, dim=0).cpu().numpy()
val_probs_cat = torch.cat(val_probs_list, dim=0).cpu().numpy()

# Metrics calculation
auroc, fpr, fnr, f2_score = metrics_calculation(val_labels_cat, val_probs_cat, THRESHOLD)

# Save results
results = {"Threshold": THRESHOLD, "AUROC": auroc, "FPR": fpr, "FNR": fnr, "F2-score": f2_score}
with open(os.path.join(exp_dir, 'test.json'), 'w') as f:
    json.dump(results, f, indent=4)

print(f'Threshold: {THRESHOLD}')
print(f"AUROC: {auroc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, F2-score: {f2_score:.4f}")


# Test AVA (SNR)
snr_list = [10, 5, 0, -5, -10]
results_snr = {}

for snr in snr_list:
    print(f'Loading AVA SNR={snr} test set ...')
    test_dataset = AVA_Tuple(
        f'/share/nas169/jethrowang/TinyVAD/data/AVA/snr/{snr}',
        max_duration=max_duration,
        sample_duration=WINDOW_SIZE,
        overlap=overlap,
        feature_extraction=(not SINC_CONV),
    )
    print(f"Test dataset size: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Variables for storing predictions and true labels
    val_probs_list, val_labels_list = [], []

    # Test loop
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'SNR={snr} Testing {name}'):
            if SINC_CONV:
                val_inputs = [item[0].to(device) for item in batch]
            else:
                val_inputs = [item[1].to(device) for item in batch]
            val_labels = [item[2].to(device).float().unsqueeze(1) for item in batch]
            val_inputs = torch.cat(val_inputs, dim=0)
            val_labels = torch.cat(val_labels, dim=0)

            val_probs = model.predict(val_inputs)

            # Apply median smoothing filter
            val_probs_list, val_labels_list = median_smoothing_filter(val_probs, val_labels, val_probs_list, val_labels_list, median_kernel_size, device)

    # Concatenate results
    val_labels_cat = torch.cat(val_labels_list, dim=0).cpu().numpy()
    val_probs_cat = torch.cat(val_probs_list, dim=0).cpu().numpy()

    # Metrics calculation
    auroc, fpr, fnr, f2_score = metrics_calculation(val_labels_cat, val_probs_cat, THRESHOLD)

    # Save results
    results_snr[snr] = {'AUROC': auroc, 'FPR': fpr, 'FNR': fnr, 'F2-score': f2_score}

    print(f"SNR={snr}, AUROC: {auroc:.4f}, FPR: {fpr:.4f}, FNR: {fnr:.4f}, F2-score: {f2_score:.4f}")

# Save results
with open(os.path.join(exp_dir, 'test_snr.json'), 'w') as f:
    json.dump(results_snr, f, indent=4)
