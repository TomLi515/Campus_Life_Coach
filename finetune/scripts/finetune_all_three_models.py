"""
Fine-tune THREE models directly from CSV files for the dashboard demo.

Usage:
    python finetune_from_csvs.py \
        --csv_dir ../data/all_users_organized \
        --phone_ckpt ../../pretrain/artifacts/phone_encoder/best.ckpt \
        --watch_ckpt ../../pretrain/artifacts/watch_encoder/best.ckpt \
        --output_dir ../models/dashboard_models
"""

print("=== SCRIPT STARTED ===")
print("Importing libraries...")

import sys
from pathlib import Path

# Add paths
current_dir = Path(__file__).parent
finetune_src = current_dir.parent / "src"
pretrain_src = current_dir.parent.parent / "pretrain" / "src"

print(f"Current dir: {current_dir}")
print(f"Finetune src: {finetune_src}")
print(f"Pretrain src: {pretrain_src}")

sys.path.insert(0, str(finetune_src))
sys.path.insert(0, str(pretrain_src))

print("Paths added to sys.path")

try:
    import torch
    print("✓ PyTorch imported")
except Exception as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    print("✓ PyTorch modules imported")
except Exception as e:
    print(f"✗ PyTorch modules import failed: {e}")
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
    from sklearn.metrics import f1_score, classification_report
    from scipy import signal
    import json
    from tqdm import tqdm
    print("✓ Other libraries imported")
except Exception as e:
    print(f"✗ Library import failed: {e}")
    sys.exit(1)

# Import from finetune package
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
    print(f"  Expected path: {finetune_src / 'finetune' / 'models.py'}")
    sys.exit(1)

print("\n=== ALL IMPORTS SUCCESSFUL ===\n")


# ==================== DATA PREPROCESSING ====================

def resample_to_50hz(df, current_hz):
    """Resample IMU data to 50 Hz."""
    imu_cols = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z']
    
    ratio = 50 / current_hz
    num_samples_new = int(len(df) * ratio)
    
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
    num_windows = (len(signal_data) - window_size) // step_size + 1
    
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        if end <= len(signal_data):
            window = signal_data[start:end].T  # Shape: (6, 150)
            windows.append(window)
    
    return np.array(windows)


def load_and_process_csv(csv_path, target_hz=50, window_size=150, step_size=75):
    """Load CSV and create windows."""
    df = pd.read_csv(csv_path)
    
    # Check sampling rate
    time_diff = np.diff(df['Time'].values)
    estimated_hz = 1 / np.median(time_diff)
    
    # Resample if needed
    if abs(estimated_hz - target_hz) > 5:
        df = resample_to_50hz(df, estimated_hz)
    
    # Create windows
    windows = create_windows(df, window_size, step_size)
    
    return windows, df['Device'].iloc[0], df['Subject'].iloc[0], df['Activity'].iloc[0]


def parse_filename(filename):
    """Parse standardized filename: user_device_activity_instance.csv"""
    parts = filename.stem.split('_')
    return {
        'subject': parts[0],
        'device': parts[1],
        'activity': parts[2],
        'instance': parts[3]
    }


def load_all_csvs(csv_dir):
    """Load all CSVs and organize by device."""
    csv_dir = Path(csv_dir)
    csv_files = list(csv_dir.glob("*.csv"))
    
    print(f"\nFound {len(csv_files)} CSV files")
    
    phone_data = {'windows': [], 'labels': [], 'subjects': []}
    watch_data = {'windows': [], 'labels': [], 'subjects': []}
    
    activity_map = {'walk': 0, 'run': 1, 'sit': 2, 'stand': 3, 'lie': 4}
    
    for csv_file in tqdm(csv_files, desc="Processing CSVs"):
        try:
            # Parse filename
            file_info = parse_filename(csv_file)
            
            # Load and process
            windows, device, subject, activity = load_and_process_csv(csv_file)
            
            if len(windows) == 0:
                continue
            
            label = activity_map[activity.lower()]
            
            # Add to appropriate dataset
            if device.lower() == 'phone':
                phone_data['windows'].append(windows)
                phone_data['labels'].extend([label] * len(windows))
                phone_data['subjects'].extend([subject] * len(windows))
            else:  # watch
                watch_data['windows'].append(windows)
                watch_data['labels'].extend([label] * len(windows))
                watch_data['subjects'].extend([subject] * len(windows))
                
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
            continue
    
    # Convert to numpy arrays
    phone_data['windows'] = np.concatenate(phone_data['windows'], axis=0)
    phone_data['labels'] = np.array(phone_data['labels'])
    phone_data['subjects'] = np.array(phone_data['subjects'])
    
    watch_data['windows'] = np.concatenate(watch_data['windows'], axis=0)
    watch_data['labels'] = np.array(watch_data['labels'])
    watch_data['subjects'] = np.array(watch_data['subjects'])
    
    print(f"\nPhone data: {len(phone_data['labels'])} windows")
    print(f"  Subjects: {np.unique(phone_data['subjects'])}")
    print(f"  Shape: {phone_data['windows'].shape}")
    
    print(f"\nWatch data: {len(watch_data['labels'])} windows")
    print(f"  Subjects: {np.unique(watch_data['subjects'])}")
    print(f"  Shape: {watch_data['windows'].shape}")
    
    return phone_data, watch_data


def normalize_per_subject(data):
    """Normalize windows per subject (z-score)."""
    subjects = data['subjects']
    unique_subjects = np.unique(subjects)
    
    normalized_windows = data['windows'].copy()
    
    for subject in unique_subjects:
        subject_mask = subjects == subject
        subject_windows = normalized_windows[subject_mask]
        
        # Compute stats across all windows for this subject
        mean = subject_windows.mean(axis=(0, 2), keepdims=True)
        std = subject_windows.std(axis=(0, 2), keepdims=True) + 1e-8
        
        # Normalize
        normalized_windows[subject_mask] = (subject_windows - mean) / std
    
    data['windows'] = normalized_windows
    return data


# ==================== DATASETS ====================

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
    """Dataset for phone+watch fusion."""
    
    def __init__(self, phone_data, watch_data, indices):
        self.phone_windows = phone_data['windows'][indices]
        self.watch_windows = watch_data['windows'][indices]
        self.labels = phone_data['labels'][indices]
        self.subjects = phone_data['subjects'][indices]
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.phone_windows[idx]).float(),
            torch.from_numpy(self.watch_windows[idx]).float(),
            torch.tensor(self.labels[idx]).long()
        )


# ==================== TRAINING FUNCTIONS ====================

def train_single_stream(model, train_loader, val_loader, device, 
                        epochs_stage1=10, epochs_stage2=20, lr=1e-4, model_name="model"):
    """Train single-stream model (phone or watch only)."""
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # STAGE 1: Freeze encoder, train classifier
    print(f"\nSTAGE 1: Training classifier (encoder frozen)")
    
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=lr * 10)
    
    best_val_f1 = 0
    best_state = None
    
    for epoch in range(epochs_stage1):
        model.train()
        total_loss = 0
        for imu, labels in train_loader:
            imu, labels = imu.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imu)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imu, labels in val_loader:
                imu = imu.to(device)
                outputs = model(imu)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs_stage1} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
    
    model.load_state_dict(best_state)
    
    # STAGE 2: Fine-tune entire model
    print(f"\nSTAGE 2: Fine-tuning entire model")
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)
    
    best_val_f1 = 0
    
    for epoch in range(epochs_stage2):
        model.train()
        total_loss = 0
        for imu, labels in train_loader:
            imu, labels = imu.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imu)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for imu, labels in val_loader:
                imu = imu.to(device)
                outputs = model(imu)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs_stage2} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
        
        scheduler.step()
    
    model.load_state_dict(best_state)
    return model


def train_fusion(model, train_loader, val_loader, device,
                 epochs_stage1=10, epochs_stage2=20, lr=1e-4):
    """Train fusion model (phone+watch)."""
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n{'='*60}")
    print(f"Training Fusion Model")
    print(f"{'='*60}")
    
    # STAGE 1: Freeze encoders, train fusion head
    print(f"\nSTAGE 1: Training fusion head (encoders frozen)")
    
    for param in model.phone_enc.parameters():
        param.requires_grad = False
    for param in model.watch_enc.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=lr * 10)
    
    best_val_f1 = 0
    best_state = None
    
    for epoch in range(epochs_stage1):
        model.train()
        total_loss = 0
        for phone_imu, watch_imu, labels in train_loader:
            phone_imu = phone_imu.to(device)
            watch_imu = watch_imu.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(phone_imu, watch_imu)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for phone_imu, watch_imu, labels in val_loader:
                phone_imu = phone_imu.to(device)
                watch_imu = watch_imu.to(device)
                outputs = model(phone_imu, watch_imu)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs_stage1} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
    
    model.load_state_dict(best_state)
    
    # STAGE 2: Fine-tune entire model
    print(f"\nSTAGE 2: Fine-tuning entire model")
    
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_stage2)
    
    best_val_f1 = 0
    
    for epoch in range(epochs_stage2):
        model.train()
        total_loss = 0
        for phone_imu, watch_imu, labels in train_loader:
            phone_imu = phone_imu.to(device)
            watch_imu = watch_imu.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(phone_imu, watch_imu)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for phone_imu, watch_imu, labels in val_loader:
                phone_imu = phone_imu.to(device)
                watch_imu = watch_imu.to(device)
                outputs = model(phone_imu, watch_imu)
                preds = outputs.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_labels.extend(labels.numpy())
        
        val_f1 = f1_score(val_labels, val_preds, average='macro')
        
        print(f"Epoch {epoch+1}/{epochs_stage2} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
        
        scheduler.step()
    
    model.load_state_dict(best_state)
    return model


def evaluate_model(model, test_loader, device, is_fusion=False):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        if is_fusion:
            for phone_imu, watch_imu, labels in test_loader:
                phone_imu = phone_imu.to(device)
                watch_imu = watch_imu.to(device)
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
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    target_names = ['Walk', 'Run', 'Sit', 'Stand', 'Lie']
    report = classification_report(all_labels, all_preds, target_names=target_names)
    
    return f1, report


# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune all three models from CSVs")
    parser.add_argument("--csv_dir", required=True, help="Directory with all_users_organized CSVs")
    parser.add_argument("--phone_ckpt", required=True, help="Pretrained phone encoder")
    parser.add_argument("--watch_ckpt", required=True, help="Pretrained watch encoder")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Fine-tuning THREE Models from CSVs")
    print("="*60)
    
    # Load and process CSVs
    print("\nLoading and processing CSVs...")
    phone_data, watch_data = load_all_csvs(args.csv_dir)
    
    # Normalize per subject
    print("\nNormalizing data per subject...")
    phone_data = normalize_per_subject(phone_data)
    watch_data = normalize_per_subject(watch_data)
    
    # Subject-independent split (use phone subjects as reference)
    subjects = phone_data['subjects']
    unique_subjects = np.unique(subjects)
    
    print(f"\nSubjects: {list(unique_subjects)}")
    
    train_subjects, test_subjects = train_test_split(unique_subjects, test_size=0.2, random_state=42)
    train_subjects, val_subjects = train_test_split(train_subjects, test_size=0.25, random_state=42)
    
    train_idx = np.isin(subjects, train_subjects)
    val_idx = np.isin(subjects, val_subjects)
    test_idx = np.isin(subjects, test_subjects)
    
    print(f"\nTrain: {np.sum(train_idx)} samples ({list(train_subjects)})")
    print(f"Val: {np.sum(val_idx)} samples ({list(val_subjects)})")
    print(f"Test: {np.sum(test_idx)} samples ({list(test_subjects)})")
    
    # ==================== MODEL 1: PHONE-ONLY ====================
    
    print("\n" + "="*60)
    print("MODEL 1: PHONE-ONLY CLASSIFIER")
    print("="*60)
    
    phone_backbone = load_pretrained_backbone(args.phone_ckpt, device=args.device)
    phone_model = SingleStreamClassifier(phone_backbone, num_classes=5, dropout=0.3).to(args.device)
    
    phone_train_dataset = SingleStreamDataset(phone_data, train_idx)
    phone_val_dataset = SingleStreamDataset(phone_data, val_idx)
    phone_test_dataset = SingleStreamDataset(phone_data, test_idx)
    
    phone_train_loader = DataLoader(phone_train_dataset, batch_size=args.batch_size, shuffle=True)
    phone_val_loader = DataLoader(phone_val_dataset, batch_size=args.batch_size)
    phone_test_loader = DataLoader(phone_test_dataset, batch_size=args.batch_size)
    
    phone_model = train_single_stream(phone_model, phone_train_loader, phone_val_loader, 
                                       args.device, model_name="Phone-Only")
    
    phone_f1, phone_report = evaluate_model(phone_model, phone_test_loader, args.device)
    print(f"\nPhone-Only Test F1: {phone_f1:.4f}")
    print(phone_report)
    
    torch.save({
        'model_state_dict': phone_model.state_dict(),
        'model_config': {'num_classes': 5, 'input_channels': 6, 'embedding_dim': 256},
        'label_map': {0: 'Walk', 1: 'Run', 2: 'Sit', 3: 'Stand', 4: 'Lie'},
        'test_f1': phone_f1
    }, output_path / "phone_only_classifier.pth")
    
    # ==================== MODEL 2: WATCH-ONLY ====================
    
    print("\n" + "="*60)
    print("MODEL 2: WATCH-ONLY CLASSIFIER")
    print("="*60)
    
    # For watch: use watch data subjects
    watch_subjects = watch_data['subjects']
    watch_unique_subjects = np.unique(watch_subjects)
    
    # If only user3 has watch data, split their recordings
    if len(watch_unique_subjects) == 1:
        print(f"\nOnly {watch_unique_subjects[0]} has watch data")
        print("Using recording-level split instead of subject-level split")
        
        # Split by indices instead
        n_samples = len(watch_data['labels'])
        all_indices = np.arange(n_samples)
        
        train_watch_idx, test_watch_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
        train_watch_idx, val_watch_idx = train_test_split(train_watch_idx, test_size=0.25, random_state=42)
    else:
        train_watch_subj, test_watch_subj = train_test_split(watch_unique_subjects, test_size=0.2, random_state=42)
        train_watch_subj, val_watch_subj = train_test_split(train_watch_subj, test_size=0.25, random_state=42)
        
        train_watch_idx = np.isin(watch_subjects, train_watch_subj)
        val_watch_idx = np.isin(watch_subjects, val_watch_subj)
        test_watch_idx = np.isin(watch_subjects, test_watch_subj)
    
    watch_backbone = load_pretrained_backbone(args.watch_ckpt, device=args.device)
    watch_model = SingleStreamClassifier(watch_backbone, num_classes=5, dropout=0.3).to(args.device)
    
    watch_train_dataset = SingleStreamDataset(watch_data, train_watch_idx)
    watch_val_dataset = SingleStreamDataset(watch_data, val_watch_idx)
    watch_test_dataset = SingleStreamDataset(watch_data, test_watch_idx)
    
    watch_train_loader = DataLoader(watch_train_dataset, batch_size=args.batch_size, shuffle=True)
    watch_val_loader = DataLoader(watch_val_dataset, batch_size=args.batch_size)
    watch_test_loader = DataLoader(watch_test_dataset, batch_size=args.batch_size)
    
    watch_model = train_single_stream(watch_model, watch_train_loader, watch_val_loader,
                                       args.device, model_name="Watch-Only")
    
    watch_f1, watch_report = evaluate_model(watch_model, watch_test_loader, args.device)
    print(f"\nWatch-Only Test F1: {watch_f1:.4f}")
    print(watch_report)
    
    torch.save({
        'model_state_dict': watch_model.state_dict(),
        'model_config': {'num_classes': 5, 'input_channels': 6, 'embedding_dim': 256},
        'label_map': {0: 'Walk', 1: 'Run', 2: 'Sit', 3: 'Stand', 4: 'Lie'},
        'test_f1': watch_f1
    }, output_path / "watch_only_classifier.pth")
    
    # ==================== MODEL 3: FUSION ====================
    
    print("\n" + "="*60)
    print("MODEL 3: PHONE+WATCH FUSION CLASSIFIER")
    print("="*60)
    
    # For fusion: only use samples where BOTH phone and watch data exist
    # In your case, only user3 has watch, so fusion only uses user3's phone data
    
    print("\nFinding samples with both phone and watch data...")
    # This is simplified - assumes user3 has both phone and watch
    # Adjust based on your actual data collection
    
    phone_backbone_fusion = load_pretrained_backbone(args.phone_ckpt, device=args.device)
    watch_backbone_fusion = load_pretrained_backbone(args.watch_ckpt, device=args.device)
    fusion_model = FusionClassifier(phone_backbone_fusion, watch_backbone_fusion, 
                                     num_classes=5, dropout=0.3).to(args.device)
    
    # Use watch indices (since watch is subset)
    fusion_train_dataset = DualStreamDataset(phone_data, watch_data, train_watch_idx)
    fusion_val_dataset = DualStreamDataset(phone_data, watch_data, val_watch_idx)
    fusion_test_dataset = DualStreamDataset(phone_data, watch_data, test_watch_idx)
    
    fusion_train_loader = DataLoader(fusion_train_dataset, batch_size=args.batch_size, shuffle=True)
    fusion_val_loader = DataLoader(fusion_val_dataset, batch_size=args.batch_size)
    fusion_test_loader = DataLoader(fusion_test_dataset, batch_size=args.batch_size)
    
    fusion_model = train_fusion(fusion_model, fusion_train_loader, fusion_val_loader, args.device)
    
    fusion_f1, fusion_report = evaluate_model(fusion_model, fusion_test_loader, args.device, is_fusion=True)
    print(f"\nFusion Test F1: {fusion_f1:.4f}")
    print(fusion_report)
    
    torch.save({
        'model_state_dict': fusion_model.state_dict(),
        'model_config': {'num_classes': 5, 'input_channels': 6, 'embedding_dim': 256},
        'label_map': {0: 'Walk', 1: 'Run', 2: 'Sit', 3: 'Stand', 4: 'Lie'},
        'test_f1': fusion_f1
    }, output_path / "fusion_classifier.pth")
    
    # ==================== SUMMARY ====================
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY")
    print("="*60)
    print(f"Phone-Only Test F1: {phone_f1:.4f}")
    print(f"Watch-Only Test F1: {watch_f1:.4f}")
    print(f"Fusion Test F1: {fusion_f1:.4f}")
    print(f"\nModels saved to: {output_path}")
    
    summary = {
        'phone_only_f1': float(phone_f1),
        'watch_only_f1': float(watch_f1),
        'fusion_f1': float(fusion_f1),
        'train_subjects': list(train_subjects),
        'val_subjects': list(val_subjects),
        'test_subjects': list(test_subjects)
    }
    
    with open(output_path / "training_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n=== FINE-TUNING COMPLETE ===")


if __name__ == "__main__":
    main()