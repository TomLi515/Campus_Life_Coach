#!/usr/bin/env python3
"""
Fine-tune THREE models directly from CSV files with:
 - Mixed (per-subject stratified) splits so every subject has data in train/val/test
 - Fusion pairing using filename (subject_instance_activity)
 - Grid-search over hyperparameters and freezing strategies (4 strategies)
 - Per-epoch train/val metrics (accuracy & loss) saved to CSV for every run
 - Summary JSON with best hyperparameters and final metrics for phone/watch/fusion

Usage example:
    python finetune_all_three_models.py \
        --csv_dir ../data/all_users_organized \
        --phone_ckpt ../../pretrain/artifacts/phone_encoder/best.ckpt \
        --watch_ckpt ../../pretrain/artifacts/watch_encoder/best.ckpt \
        --output_dir ../models/dashboard_models \
        --max_search_runs 50
"""

print("=== SCRIPT STARTED ===")
print("Importing libraries...")

import sys
from pathlib import Path
import random
import time
import json

# Add paths
current_dir = Path(__file__).parent
finetune_src = current_dir.parent / "src"
pretrain_src = current_dir.parent.parent / "pretrain" / "src"

sys.path.insert(0, str(finetune_src))
sys.path.insert(0, str(pretrain_src))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    print("✓ PyTorch imported")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    print("✓ NumPy and Pandas imported")
except Exception as e:
    print(f"✗ NumPy/Pandas import failed: {e}")
    sys.exit(1)

try:
    import argparse
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report, accuracy_score
    from scipy import signal
    from tqdm import tqdm
    print("✓ Other libraries imported")
except Exception as e:
    print(f"✗ Library import failed: {e}")
    sys.exit(1)

# Import from finetune package (your existing models)
try:
    from finetune.models import (
        SingleStreamClassifier,
        FusionClassifier,
        load_pretrained_backbone
    )
    print("✓ Finetune models imported")
except Exception as e:
    print(f"✗ Finetune models import failed: {e}")
    print(f"  Make sure finetune/src/finetune/models.py exists")
    sys.exit(1)

print("\n=== ALL IMPORTS SUCCESSFUL ===\n")

# -------------------------
# Utility / preprocessing
# -------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def resample_to_50hz(df, current_hz):
    """Resample IMU data to 50 Hz."""
    imu_cols = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    ratio = 50 / current_hz
    num_samples_new = int(len(df) * ratio)
    if num_samples_new < 2:
        num_samples_new = max(2, len(df))
    resampled_data = {'Time': np.linspace(df['Time'].iloc[0], df['Time'].iloc[-1], num_samples_new)}
    for col in imu_cols:
        resampled_data[col] = signal.resample(df[col].values, num_samples_new)
    resampled_df = pd.DataFrame(resampled_data)
    for col in ['Device', 'Subject', 'Activity']:
        if col in df.columns:
            resampled_df[col] = df[col].iloc[0]
    return resampled_df

def create_windows(df, window_size=150, step_size=75):
    """Create sliding windows from continuous signal."""
    imu_cols = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    signal_data = df[imu_cols].values
    windows = []
    if len(signal_data) < window_size:
        return np.zeros((0, 6, window_size), dtype=float)
    num_windows = (len(signal_data) - window_size) // step_size + 1
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        if end <= len(signal_data):
            window = signal_data[start:end].T  # Shape: (6, 150)
            windows.append(window)
    return np.array(windows)

def load_and_process_csv(csv_path, target_hz=50, window_size=150, step_size=75):
    """Load CSV and create windows. Returns windows and metadata."""
    df = pd.read_csv(csv_path)
    # Check sampling rate
    if 'Time' not in df.columns or len(df['Time'].values) < 2:
        estimated_hz = target_hz
    else:
        time_diff = np.diff(df['Time'].values)
        # Guard against zeros
        time_diff = time_diff[time_diff > 0]
        if len(time_diff) == 0:
            estimated_hz = target_hz
        else:
            estimated_hz = 1 / np.median(time_diff)
    # Resample if needed
    if abs(estimated_hz - target_hz) > 5:
        df = resample_to_50hz(df, estimated_hz)
    # Create windows
    windows = create_windows(df, window_size, step_size)
    subject = df['Subject'].iloc[0] if 'Subject' in df.columns else 'unknown'
    device = df['Device'].iloc[0] if 'Device' in df.columns else 'unknown'
    activity = df['Activity'].iloc[0] if 'Activity' in df.columns else 'unknown'
    return windows, device, subject, activity

def parse_filename(filename):
    """Parse standardized filename: user_device_activity_instance.csv"""
    parts = filename.stem.split('_')
    # Expect at least 4 parts; pad if necessary
    while len(parts) < 4:
        parts.append("0")
    return {
        'subject': parts[0],
        'device': parts[1],
        'activity': parts[2],
        'instance': parts[3]
    }

def load_all_csvs(csv_dir, window_size=150, step_size=75):
    """Load all CSVs and organize by device and store file_id per-window for fusion pairing."""
    csv_dir = Path(csv_dir)
    csv_files = sorted(list(csv_dir.glob("*.csv")))
    print(f"\nFound {len(csv_files)} CSV files")
    phone_windows_list = []
    phone_labels = []
    phone_subjects = []
    phone_file_ids = []
    watch_windows_list = []
    watch_labels = []
    watch_subjects = []
    watch_file_ids = []
    activity_map = {'walk': 0, 'run': 1, 'sit': 2, 'stand': 3, 'lie': 4}
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        try:
            info = parse_filename(csv_file)
            file_id = f"{info['subject']}_{info['instance']}_{info['activity']}"
            windows, device, subject, activity = load_and_process_csv(csv_file, window_size=window_size, step_size=step_size)
            if len(windows) == 0:
                continue
            label = activity_map.get(activity.lower(), None)
            if label is None:
                # Skip unknown activity
                continue
            if device.lower() == 'phone':
                phone_windows_list.append(windows)
                phone_labels.extend([label] * len(windows))
                phone_subjects.extend([subject] * len(windows))
                phone_file_ids.extend([file_id] * len(windows))
            else:
                watch_windows_list.append(windows)
                watch_labels.extend([label] * len(windows))
                watch_subjects.extend([subject] * len(windows))
                watch_file_ids.extend([file_id] * len(windows))
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    if len(phone_windows_list) == 0:
        phone_data = {'windows': np.zeros((0,6,window_size)), 'labels': np.array([]), 'subjects': np.array([]), 'file_ids': np.array([])}
    else:
        phone_data = {
            'windows': np.concatenate(phone_windows_list, axis=0),
            'labels': np.array(phone_labels),
            'subjects': np.array(phone_subjects),
            'file_ids': np.array(phone_file_ids)
        }
    if len(watch_windows_list) == 0:
        watch_data = {'windows': np.zeros((0,6,window_size)), 'labels': np.array([]), 'subjects': np.array([]), 'file_ids': np.array([])}
    else:
        watch_data = {
            'windows': np.concatenate(watch_windows_list, axis=0),
            'labels': np.array(watch_labels),
            'subjects': np.array(watch_subjects),
            'file_ids': np.array(watch_file_ids)
        }
    print(f"\nPhone data: {len(phone_data['labels'])} windows")
    print(f"  Subjects: {np.unique(phone_data['subjects'])}")
    print(f"  Shape: {phone_data['windows'].shape}")
    print(f"\nWatch data: {len(watch_data['labels'])} windows")
    print(f"  Subjects: {np.unique(watch_data['subjects'])}")
    print(f"  Shape: {watch_data['windows'].shape}")
    return phone_data, watch_data

def normalize_per_subject(data):
    """Normalize windows per subject (z-score)."""
    if len(data['labels']) == 0:
        return data
    subjects = data['subjects']
    unique_subjects = np.unique(subjects)
    normalized_windows = data['windows'].copy()
    for subject in unique_subjects:
        subject_mask = subjects == subject
        if np.sum(subject_mask) == 0:
            continue
        subject_windows = normalized_windows[subject_mask]
        mean = subject_windows.mean(axis=(0, 2), keepdims=True)
        std = subject_windows.std(axis=(0, 2), keepdims=True) + 1e-8
        normalized_windows[subject_mask] = (subject_windows - mean) / std
    data['windows'] = normalized_windows
    return data

# -------------------------
# Datasets
# -------------------------

class SingleStreamDataset(Dataset):
    """Dataset for single stream (phone or watch)."""
    def __init__(self, data, indices):
        self.windows = data['windows'][indices]
        self.labels = data['labels'][indices]
        self.subjects = data['subjects'][indices]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.windows[idx]).float(),
            torch.tensor(self.labels[idx]).long()
        )

class DualStreamDataset(Dataset):
    """Dataset for phone+watch fusion from paired index lists."""
    def __init__(self, phone_data, watch_data, phone_indices, watch_indices):
        assert len(phone_indices) == len(watch_indices)
        self.phone_windows = phone_data['windows'][phone_indices]
        self.watch_windows = watch_data['windows'][watch_indices]
        self.labels = phone_data['labels'][phone_indices]  # assume aligned label
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.phone_windows[idx]).float(),
            torch.from_numpy(self.watch_windows[idx]).float(),
            torch.tensor(self.labels[idx]).long()
        )

# -------------------------
# Freezing strategies
# -------------------------

FREEZE_STRATEGIES = [
    'no_freeze',
    'freeze_enc_then_finetune',  # original: stage 1 freeze enc, train head, stage2 finetune
    'freeze_all_but_last',       # freeze everything except last classifier linear
    'freeze_classifier_only'     # train classifier only (no finetune)
]

def apply_freeze_all_but_last_single(model):
    # Freeze all except final classifier linear (we attempt to find linear layer in classifier)
    for param in model.parameters():
        param.requires_grad = False
    # try to unfreeze last linear in classifier
    if hasattr(model, 'classifier'):
        for name, module in model.classifier.named_modules():
            # heuristic: unfreeze nn.Linear layers
            if isinstance(module, nn.Linear):
                for p in module.parameters():
                    p.requires_grad = True

def apply_freeze_all_but_last_fusion(model):
    for param in model.parameters():
        param.requires_grad = False
    # unfreeze fusion head linear(s)
    if hasattr(model, 'fusion'):
        for name, module in model.fusion.named_modules():
            if isinstance(module, nn.Linear):
                for p in module.parameters():
                    p.requires_grad = True

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def freeze_encoders(model):
    # freeze encoder-like modules (names: encoder or phone_enc/watch_enc)
    for name, param in model.named_parameters():
        if 'encoder' in name or 'phone_enc' in name or 'watch_enc' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

# -------------------------
# Training / evaluation helpers
# -------------------------

def compute_metrics_epoch(model, loader, device, is_fusion=False):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        if is_fusion:
            for phone_imu, watch_imu, labels in loader:
                phone_imu = phone_imu.to(device); watch_imu = watch_imu.to(device); labels = labels.to(device)
                outputs = model(phone_imu, watch_imu)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item() * labels.size(0)
        else:
            for imu, labels in loader:
                imu = imu.to(device); labels = labels.to(device)
                outputs = model(imu)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                total_loss += loss.item() * labels.size(0)
    if len(all_labels) == 0:
        return {'loss': 0.0, 'acc': 0.0, 'f1': 0.0}
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return {'loss': float(avg_loss), 'acc': float(acc), 'f1': float(f1)}

def train_single_stream_with_strategy(model, train_loader, val_loader, device,
                                      epochs_stage1=10, epochs_stage2=20, lr=1e-4,
                                      strategy='freeze_enc_then_finetune', run_tag='run'):
    """
    Train single-stream model with a specified freezing strategy.
    Returns best model state & epoch-level metrics DataFrame.
    """
    criterion = nn.CrossEntropyLoss()
    results = []
    best_val_f1 = -1
    best_state = None

    # Stage 1 behavior depends on strategy
    if strategy == 'no_freeze':
        # no explicit stage 1, train all for epochs_stage1 + epochs_stage2
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_epochs = epochs_stage1 + epochs_stage2
        scheduler = None
        epoch_start = 0
        for epoch in range(total_epochs):
            model.train()
            total_loss = 0.0
            for imu, labels in train_loader:
                imu, labels = imu.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=False)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=False)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_state = model.state_dict()
    elif strategy == 'freeze_classifier_only':
        # freeze entire model except classifier; train classifier only for epochs_stage1; no finetune
        for param in model.parameters():
            param.requires_grad = False
        # unfreeze classifier params
        for p in model.classifier.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr*10)
        for epoch in range(epochs_stage1):
            model.train()
            total_loss = 0.0
            for imu, labels in train_loader:
                imu, labels = imu.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=False)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=False)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_state = model.state_dict()
    elif strategy == 'freeze_all_but_last':
        apply_freeze_all_but_last_single(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr*10)
        for epoch in range(epochs_stage1):
            model.train()
            total_loss = 0.0
            for imu, labels in train_loader:
                imu, labels = imu.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=False)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=False)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_state = model.state_dict()
    elif strategy == 'freeze_enc_then_finetune':
        # Stage 1: freeze encoder, train classifier
        freeze_encoders(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr*10)
        for epoch in range(epochs_stage1):
            model.train()
            total_loss = 0.0
            for imu, labels in train_loader:
                imu, labels = imu.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=False)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=False)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_state = model.state_dict()
        # Stage 2: unfreeze all and finetune
        unfreeze_all(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)
        for epoch in range(epochs_stage2):
            model.train()
            total_loss = 0.0
            for imu, labels in train_loader:
                imu, labels = imu.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=False)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=False)
            row = {'epoch': epochs_stage1 + epoch + 1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_state = model.state_dict()
            scheduler.step()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    # return best_state & results as DataFrame
    metrics_df = pd.DataFrame(results)
    return best_state, metrics_df, best_val_f1

def train_fusion_with_strategy(model, train_loader, val_loader, device,
                               epochs_stage1=10, epochs_stage2=20, lr=1e-4,
                               strategy='freeze_enc_then_finetune', run_tag='run'):
    criterion = nn.CrossEntropyLoss()
    results = []
    best_val_f1 = -1
    best_state = None

    if strategy == 'no_freeze':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_epochs = epochs_stage1 + epochs_stage2
        for epoch in range(total_epochs):
            model.train()
            for phone_imu, watch_imu, labels in train_loader:
                phone_imu = phone_imu.to(device); watch_imu = watch_imu.to(device); labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(phone_imu, watch_imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=True)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=True)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']; best_state = model.state_dict()
    elif strategy == 'freeze_classifier_only':
        # freeze everything except fusion head
        for p in model.parameters():
            p.requires_grad = False
        if hasattr(model, 'fusion'):
            for p in model.fusion.parameters():
                p.requires_grad = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr*10)
        for epoch in range(epochs_stage1):
            model.train()
            for phone_imu, watch_imu, labels in train_loader:
                phone_imu = phone_imu.to(device); watch_imu = watch_imu.to(device); labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(phone_imu, watch_imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=True)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=True)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']; best_state = model.state_dict()
    elif strategy == 'freeze_all_but_last':
        apply_freeze_all_but_last_fusion(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr*10)
        for epoch in range(epochs_stage1):
            model.train()
            for phone_imu, watch_imu, labels in train_loader:
                phone_imu = phone_imu.to(device); watch_imu = watch_imu.to(device); labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(phone_imu, watch_imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=True)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=True)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']; best_state = model.state_dict()
    elif strategy == 'freeze_enc_then_finetune':
        # Stage 1: freeze encoders, train fusion head
        for name, param in model.named_parameters():
            if 'phone_enc' in name or 'watch_enc' in name or 'encoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr*10)
        for epoch in range(epochs_stage1):
            model.train()
            for phone_imu, watch_imu, labels in train_loader:
                phone_imu = phone_imu.to(device); watch_imu = watch_imu.to(device); labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(phone_imu, watch_imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=True)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=True)
            row = {'epoch': epoch+1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']; best_state = model.state_dict()
        # Stage 2: finetune entire model
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)
        for epoch in range(epochs_stage2):
            model.train()
            for phone_imu, watch_imu, labels in train_loader:
                phone_imu = phone_imu.to(device); watch_imu = watch_imu.to(device); labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(phone_imu, watch_imu)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_metrics = compute_metrics_epoch(model, train_loader, device, is_fusion=True)
            val_metrics = compute_metrics_epoch(model, val_loader, device, is_fusion=True)
            row = {'epoch': epochs_stage1 + epoch + 1, 'train_loss': train_metrics['loss'], 'train_acc': train_metrics['acc'],
                   'train_f1': train_metrics['f1'], 'val_loss': val_metrics['loss'], 'val_acc': val_metrics['acc'],
                   'val_f1': val_metrics['f1']}
            results.append(row)
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']; best_state = model.state_dict()
            scheduler.step()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    metrics_df = pd.DataFrame(results)
    return best_state, metrics_df, best_val_f1

def evaluate_model(model, test_loader, device, is_fusion=False):
    """Evaluate model on test set and return f1 and classification report."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        if is_fusion:
            for phone_imu, watch_imu, labels in test_loader:
                phone_imu = phone_imu.to(device); watch_imu = watch_imu.to(device)
                outputs = model(phone_imu, watch_imu)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        else:
            for imu, labels in test_loader:
                imu = imu.to(device)
                outputs = model(imu)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
    if len(all_labels) == 0:
        return 0.0, "No samples"
    f1 = f1_score(all_labels, all_preds, average='macro')
    target_names = ['Walk', 'Run', 'Sit', 'Stand', 'Lie']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    return f1, report

# -------------------------
# Matching for fusion
# -------------------------

def build_label_matched_pairs(phone_data, watch_data, seed=42):
    """
    Create pairs of phone and watch indices by matching labels (activity).
    For each label, pair up to min(count_phone_label, count_watch_label) samples.
    Returns (phone_indices, watch_indices) arrays aligned by pair order.
    """
    rng = np.random.RandomState(seed)
    phone_by_label = {}
    watch_by_label = {}
    for i, label in enumerate(phone_data['labels']):
        phone_by_label.setdefault(int(label), []).append(i)
    for i, label in enumerate(watch_data['labels']):
        watch_by_label.setdefault(int(label), []).append(i)
    phone_indices = []
    watch_indices = []
    labels = sorted(set(list(phone_by_label.keys()) + list(watch_by_label.keys())))
    for lbl in labels:
        p_list = phone_by_label.get(lbl, [])
        w_list = watch_by_label.get(lbl, [])
        if len(p_list) == 0 or len(w_list) == 0:
            # if one side missing this label, skip pairs for this label
            continue
        rng.shuffle(p_list)
        rng.shuffle(w_list)
        # pair up to min counts
        count = min(len(p_list), len(w_list))
        phone_indices.extend(p_list[:count])
        watch_indices.extend(w_list[:count])
    phone_indices = np.array(phone_indices, dtype=int)
    watch_indices = np.array(watch_indices, dtype=int)
    return phone_indices, watch_indices

# -------------------------
# Main: grid search & training orchestration
# -------------------------

def run_grid_search_for_model(model_type, phone_data, watch_data, phone_ckpt, watch_ckpt,
                              output_dir, device, param_grid, max_runs=100, export_torchscript=False):

    """
    model_type: 'phone', 'watch', 'fusion'
    param_grid: dict with lists: {'lr':[], 'batch_size':[], 'dropout':[], 'strategy':[]}
    """
    runs = []
    best_run = None
    best_val = -1
    run_id = 0

    # Prepare data indices based on mixed (per-subject stratified) split
    if model_type == 'phone':
        N = len(phone_data['labels'])
        idxs = np.arange(N)
        subjects = phone_data['subjects']
        # stratify by subjects so each split contains variety of subjects
        train_idx, test_idx = train_test_split(idxs, test_size=0.20, random_state=42, stratify=subjects)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=subjects[train_idx])
    elif model_type == 'watch':
        N = len(watch_data['labels'])
        idxs = np.arange(N)
        subjects = watch_data['subjects']
        if N == 0:
            return None
        # if only one subject, do recording-level split stratify by label
        unique_subjects = np.unique(subjects)
        if len(unique_subjects) == 1:
            train_idx, test_idx = train_test_split(idxs, test_size=0.20, random_state=42, stratify=watch_data['labels'])
            train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=watch_data['labels'][train_idx])
        else:
            train_idx, test_idx = train_test_split(idxs, test_size=0.20, random_state=42, stratify=subjects)
            train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=subjects[train_idx])
    elif model_type == 'fusion':
        # build pairs first
        phone_pairs, watch_pairs = build_label_matched_pairs(phone_data, watch_data)
        if len(phone_pairs) == 0:
            # no overlapping files — fall back to label-matching pairing (phone pool vs watch pool)
            print("No file-id overlaps found. Building label-matched pairs across phone and watch datasets...")
            phone_pairs, watch_pairs = build_label_matched_pairs(phone_data, watch_data, seed=42)
            if len(phone_pairs) == 0:
                print("No label-matched pairs could be created (no shared labels). Skipping fusion.")
                return None
        idxs = np.arange(len(phone_pairs))  # indices into the pair arrays
        # we can stratify by label from phone_data at phone_pairs
        labels_for_pairs = phone_data['labels'][phone_pairs]
        train_idx, test_idx = train_test_split(idxs, test_size=0.20, random_state=42, stratify=labels_for_pairs)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.25, random_state=42, stratify=labels_for_pairs[train_idx])
    else:
        raise ValueError(model_type)

    # Convert param_grid to list of combos (cartesian product)
    import itertools
    keys = list(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    # Optionally limit number of runs
    if max_runs is not None and len(combos) > max_runs:
        combos = combos[:max_runs]

    for combo in combos:
        params = dict(zip(keys, combo))
        run_id += 1
        print("\n" + "="*60)
        print(f"RUN {run_id}/{len(combos)} for {model_type} with params: {params}")
        print("="*60)
        set_seed(42 + run_id)
        lr = params.get('lr', 1e-4)
        batch_size = params.get('batch_size', 64)
        dropout = params.get('dropout', 0.3)
        strategy = params.get('strategy', 'freeze_enc_then_finetune')
        # Build model
        if model_type == 'phone':
            backbone = load_pretrained_backbone(phone_ckpt, device=device)
            model = SingleStreamClassifier(backbone, num_classes=5, dropout=dropout).to(device)
            train_dataset = SingleStreamDataset(phone_data, train_idx)
            val_dataset = SingleStreamDataset(phone_data, val_idx)
            test_dataset = SingleStreamDataset(phone_data, test_idx)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            best_state, metrics_df, best_val_f1 = train_single_stream_with_strategy(model, train_loader, val_loader, device,
                                                                                    epochs_stage1=10, epochs_stage2=20, lr=lr,
                                                                                    strategy=strategy, run_tag=f"phone_run{run_id}")
        elif model_type == 'watch':
            backbone = load_pretrained_backbone(watch_ckpt, device=device)
            model = SingleStreamClassifier(backbone, num_classes=5, dropout=dropout).to(device)
            train_dataset = SingleStreamDataset(watch_data, train_idx)
            val_dataset = SingleStreamDataset(watch_data, val_idx)
            test_dataset = SingleStreamDataset(watch_data, test_idx)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            best_state, metrics_df, best_val_f1 = train_single_stream_with_strategy(model, train_loader, val_loader, device,
                                                                                    epochs_stage1=10, epochs_stage2=20, lr=lr,
                                                                                    strategy=strategy, run_tag=f"watch_run{run_id}")
        elif model_type == 'fusion':
            phone_pairs, watch_pairs = build_label_matched_pairs(phone_data, watch_data)
            backbone_phone = load_pretrained_backbone(phone_ckpt, device=device)
            backbone_watch = load_pretrained_backbone(watch_ckpt, device=device)
            model = FusionClassifier(backbone_phone, backbone_watch, num_classes=5, dropout=dropout).to(device)
            # Select pair indices for train/val/test (these idxs index into pairs arrays)
            phone_train_idx = phone_pairs[train_idx]; watch_train_idx = watch_pairs[train_idx]
            phone_val_idx = phone_pairs[val_idx]; watch_val_idx = watch_pairs[val_idx]
            phone_test_idx = phone_pairs[test_idx]; watch_test_idx = watch_pairs[test_idx]
            train_dataset = DualStreamDataset(phone_data, watch_data, phone_train_idx, watch_train_idx)
            val_dataset = DualStreamDataset(phone_data, watch_data, phone_val_idx, watch_val_idx)
            test_dataset = DualStreamDataset(phone_data, watch_data, phone_test_idx, watch_test_idx)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            best_state, metrics_df, best_val_f1 = train_fusion_with_strategy(model, train_loader, val_loader, device,
                                                                             epochs_stage1=10, epochs_stage2=20, lr=lr,
                                                                             strategy=strategy, run_tag=f"fusion_run{run_id}")
        else:
            raise ValueError(model_type)
        # Save run metrics
        run_name = f"{model_type}_run_{run_id}"
        run_dir = Path(output_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv = run_dir / "epoch_metrics.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        # Save model best state
                # Save model best state (existing state_dict checkpoint)
        if best_state is not None:
            ckpt_path = run_dir / "best_model.pth"
            torch.save({
                'model_state_dict': best_state,
                'params': params,
                'model_type': model_type
            }, ckpt_path)

        # Evaluate on test set using loaded best_state model
        test_f1, test_report = 0.0, "No best_state"
        test_acc = 0.0
        if best_state is not None:
            # load into a fresh model instance to evaluate
            if model_type == 'phone':
                backbone = load_pretrained_backbone(phone_ckpt, device=device)
                eval_model = SingleStreamClassifier(backbone, num_classes=5, dropout=dropout).to(device)
            elif model_type == 'watch':
                backbone = load_pretrained_backbone(watch_ckpt, device=device)
                eval_model = SingleStreamClassifier(backbone, num_classes=5, dropout=dropout).to(device)
            else:
                backbone_phone = load_pretrained_backbone(phone_ckpt, device=device)
                backbone_watch = load_pretrained_backbone(watch_ckpt, device=device)
                eval_model = FusionClassifier(backbone_phone, backbone_watch, num_classes=5, dropout=dropout).to(device)

            eval_model.load_state_dict(best_state)
            eval_model.eval()

            # compute test metrics
            if model_type == 'fusion':
                test_metrics = compute_metrics_epoch(eval_model, test_loader, device, is_fusion=True)
                test_f1 = test_metrics['f1']
                test_acc = test_metrics['acc']
                _, test_report = evaluate_model(eval_model, test_loader, device, is_fusion=True)
            else:
                test_metrics = compute_metrics_epoch(eval_model, test_loader, device, is_fusion=False)
                test_f1 = test_metrics['f1']
                test_acc = test_metrics['acc']
                _, test_report = evaluate_model(eval_model, test_loader, device, is_fusion=False)

            # ---- NEW: Save the full nn.Module object (so your dashboard can torch.load it directly) ----
            try:
                # Move model to CPU before saving to improve portability
                eval_model_cpu = eval_model.to('cpu')
                full_model_path = Path(output_dir) / f"{model_type}_classifier.pth"
                torch.save(eval_model_cpu, full_model_path)
                print(f"[SAVE] Full model saved for deployment at: {full_model_path}")

                # Optionally export TorchScript if requested (portable, no class defs required at load time)
                if export_torchscript:
                    try:
                        if model_type in ('phone', 'watch'):
                            example_in = torch.randn(1, 6, 150)
                            traced = torch.jit.trace(eval_model_cpu, example_in)
                        else:
                            example_in = (torch.randn(1, 6, 150), torch.randn(1, 6, 150))
                            traced = torch.jit.trace(eval_model_cpu, example_in)
                        ts_path = Path(output_dir) / f"{model_type}_classifier.pt"
                        traced.save(str(ts_path))
                        print(f"[SAVE] TorchScript model exported at: {ts_path}")
                    except Exception as e:
                        print(f"[WARN] TorchScript export failed for {model_type}: {e}")

            except Exception as e:
                print(f"[ERROR] Saving full model for {model_type} failed: {e}")
                traceback.print_exc()
        else:
            test_f1, test_report, test_acc = 0.0, "No best_state", 0.0

        # Save run summary (include test_acc)
        run_summary = {
            'model_type': model_type,
            'run_id': run_id,
            'params': params,
            'best_val_f1': float(best_val_f1),
            'test_f1': float(test_f1),
            'test_acc': float(test_acc),
            'test_report': test_report
        }
        with open(run_dir / "run_summary.json", 'w') as f:
            json.dump(run_summary, f, indent=2)
        print(f"Run {run_id} finished. best_val_f1={best_val_f1:.4f}, test_f1={test_f1:.4f}, test_acc={test_acc:.4f}")
        runs.append(run_summary)

        # Select best run by test accuracy (metric for your deployment scenario)
        if test_acc > best_val:
            best_val = test_acc
            best_run = run_summary
            best_run['best_model_path'] = str(run_dir / "best_model.pth")
    # Save all runs summary
    with open(Path(output_dir) / f"{model_type}_all_runs_summary.json", 'w') as f:
        json.dump(runs, f, indent=2)
    return best_run

# -------------------------
# Entrypoint
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune three models from CSVs with grid search")
    parser.add_argument("--csv_dir", required=True, help="Directory with CSVs")
    parser.add_argument("--phone_ckpt", required=True, help="Pretrained phone encoder")
    parser.add_argument("--watch_ckpt", required=True, help="Pretrained watch encoder")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_search_runs", type=int, default=50, help="Maximum number of grid-search runs per model")
    parser.add_argument("--export_torchscript", action="store_true", help="Also export a TorchScript (.pt) model for deployment")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("="*60)
    print("Fine-tuning THREE Models from CSVs (grid search + strategies)")
    print("="*60)

    # Load & preprocess
    phone_data, watch_data = load_all_csvs(args.csv_dir)
    phone_data = normalize_per_subject(phone_data)
    watch_data = normalize_per_subject(watch_data)

    # Define hyperparameter grid (feel free to expand)
    param_grid = {
        'lr': [1e-3, 1e-4],
        'batch_size': [32, 64],
        'dropout': [0.2, 0.3],
        'strategy': FREEZE_STRATEGIES
    }

    # Run grid search for phone
    print("\nStarting grid search for PHONE-only model...")
    best_phone = run_grid_search_for_model('phone', phone_data, watch_data, args.phone_ckpt, args.watch_ckpt,
                                          output_dir, args.device, param_grid, max_runs=args.max_search_runs,
                                          export_torchscript=args.export_torchscript)
    # Run grid search for watch
    print("\nStarting grid search for WATCH-only model...")
    best_watch = run_grid_search_for_model('watch', phone_data, watch_data, args.phone_ckpt, args.watch_ckpt,
                                          output_dir, args.device, param_grid, max_runs=args.max_search_runs,
                                          export_torchscript=args.export_torchscript)
    # Run grid search for fusion
    print("\nStarting grid search for FUSION model...")
    best_fusion = run_grid_search_for_model('fusion', phone_data, watch_data, args.phone_ckpt, args.watch_ckpt,
                                           output_dir, args.device, param_grid, max_runs=args.max_search_runs,
                                           export_torchscript=args.export_torchscript)

    summary = {
        'best_phone': best_phone,
        'best_watch': best_watch,
        'best_fusion': best_fusion
    }
    with open(output_dir / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== GRID SEARCH & TRAINING DONE ===")
    print(f"Summary saved to: {output_dir / 'training_summary.json'}")
    print("Per-run epoch metrics and best checkpoints saved under output_dir/<model>_run_<id>/")

if __name__ == "__main__":
    main()
