"""
Finetune models that use pretrained backbones.
Place this file at: finetune/src/finetune/models.py
"""

import torch
import torch.nn as nn
from pretrain.models.backbones import build_backbone
from pretrain.models.model import EncoderWithHead
import pathlib
import platform

# Fix for loading PosixPath checkpoints on Windows
if platform.system() == 'Windows':
    import pathlib
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath


def load_pretrained_backbone(ckpt_path, device='cpu'):
    """
    Load pretrained encoder backbone from checkpoint.
    
    Args:
        ckpt_path: Path to pretrained .ckpt file
        device: Device to load model on
        
    Returns:
        Pretrained backbone (without classification head)
    """
    print(f"Loading pretrained backbone from {ckpt_path}")
    
    # Load checkpoint with weights_only=False for compatibility
    # This is safe since we're loading our own pretrained checkpoints
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    print(f"Checkpoint keys: {list(ckpt.keys())}")
    
    # Extract config and state_dict based on checkpoint format
    if 'model_state' in ckpt:
        # Custom checkpoint format
        state_dict = ckpt['model_state']
        config = ckpt.get('config', {})
    elif 'hyper_parameters' in ckpt:
        # Lightning checkpoint format
        config = ckpt['hyper_parameters']
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        # Standard PyTorch format
        state_dict = ckpt['model_state_dict']
        config = ckpt.get('model_config', {})
    else:
        # Direct state dict
        state_dict = ckpt
        config = {}
    
    print(f"Config keys: {list(config.keys()) if isinstance(config, dict) else 'N/A'}")
    
    # Extract model config if nested
    if 'model' in config and isinstance(config['model'], dict):
        model_config = config['model']
    else:
        model_config = config
    
    # Extract hyperparameters with defaults
    # Map possible config key names
    backbone_name = model_config.get('backbone', model_config.get('backbone_name', 'convnet_small'))
    input_channels = model_config.get('input_channels', model_config.get('in_channels', 6))
    embedding_dim = model_config.get('embedding_dim', model_config.get('hidden_dim', 256))
    num_classes = model_config.get('num_classes', 6)
    dropout = model_config.get('dropout', 0.1)
    
    print(f"  Backbone from config: {backbone_name}")
    
    # Map old backbone names to new ones if needed
    backbone_mapping = {
        'deep_conv_lstm': 'convnet_small',
        'resnet1d': 'convnet_small',
        'simple_cnn': 'convnet_small',
        'inception': 'inception_like',
        'transformer': 'transformer_small'
    }
    
    if backbone_name in backbone_mapping:
        old_name = backbone_name
        backbone_name = backbone_mapping[backbone_name]
        print(f"  Mapped '{old_name}' → '{backbone_name}'")
    
    print(f"  Input channels: {input_channels}")
    print(f"  Embedding dim: {embedding_dim}")
    
    # Recreate encoder (with head, as that's how it was saved)
    encoder_with_head = EncoderWithHead(
        backbone_name=backbone_name,
        input_channels=input_channels,
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Load state dict (handle different prefixes)
    try:
        encoder_with_head.load_state_dict(state_dict, strict=False)
        print(f"  ✓ Loaded state dict")
    except RuntimeError as e:
        print(f"  ⚠ Error loading state dict: {e}")
        # Try removing prefixes if present
        new_state_dict = {}
        for k, v in state_dict.items():
            # Remove 'model.' or 'encoder.' prefix if present
            new_key = k.replace('model.', '').replace('encoder.', '')
            new_state_dict[new_key] = v
        encoder_with_head.load_state_dict(new_state_dict, strict=False)
        print(f"  ✓ Loaded with prefix adjustment")
    
    # Extract just the backbone (without the classification head)
    backbone = encoder_with_head.backbone
    
    print(f"  ✓ Loaded backbone successfully")
    
    return backbone


class SingleStreamClassifier(nn.Module):
    """
    Single-stream classifier (phone OR watch).
    Uses pretrained backbone + new classification head.
    """
    
    def __init__(self, pretrained_backbone, num_classes, dropout=0.3):
        super().__init__()
        
        self.encoder = pretrained_backbone
        self.embedding_dim = 256  # Standard embedding dim from pretrained models
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embedding_dim),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, 6, 150) - IMU windows
            
        Returns:
            logits: (batch, num_classes)
        """
        embeddings = self.encoder(x)  # (batch, 256)
        logits = self.classifier(embeddings)  # (batch, num_classes)
        return logits


class FusionClassifier(nn.Module):
    """
    Dual-stream fusion classifier (phone AND watch).
    Combines embeddings from both pretrained backbones.
    """
    
    def __init__(self, pretrained_phone_backbone, pretrained_watch_backbone, 
                 num_classes, dropout=0.3):
        super().__init__()
        
        self.phone_enc = pretrained_phone_backbone
        self.watch_enc = pretrained_watch_backbone
        self.embedding_dim = 256  # Each encoder outputs 256-dim
        
        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.embedding_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(self.embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, phone_x, watch_x):
        """
        Args:
            phone_x: (batch, 6, 150) - Phone IMU windows
            watch_x: (batch, 6, 150) - Watch IMU windows
            
        Returns:
            logits: (batch, num_classes)
        """
        phone_emb = self.phone_enc(phone_x)  # (batch, 256)
        watch_emb = self.watch_enc(watch_x)  # (batch, 256)
        
        # Concatenate embeddings
        combined = torch.cat([phone_emb, watch_emb], dim=1)  # (batch, 512)
        
        # Fusion classification
        logits = self.fusion(combined)  # (batch, num_classes)
        return logits


__all__ = ['SingleStreamClassifier', 'FusionClassifier', 'load_pretrained_backbone']