#!/usr/bin/env python3
"""
Extract the model_config from checkpoint files to understand architecture
"""

import torch
from pathlib import Path
import json

MODEL_DIR = Path(r"C:\Users\RAJ DAVE\Desktop\thesis\code\new_dataset_folder_cloned\Campus_coach_MLS\final_repo\Campus_Life_Coach\finetune\models\dashboard_models")

files_to_check = [
    "phone_only_classifier.pth",
    "watch_only_classifier.pth", 
    "fusion_classifier.pth"
]

print("=" * 80)
print("EXTRACTING MODEL CONFIG AND STATE DICT DETAILS")
print("=" * 80 + "\n")

for filename in files_to_check:
    filepath = MODEL_DIR / filename
    
    print(f"\n{'='*80}")
    print(f"FILE: {filename}")
    print(f"{'='*80}\n")
    
    try:
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        
        # Extract config
        model_config = checkpoint.get('model_config', {})
        label_map = checkpoint.get('label_map', {})
        test_f1 = checkpoint.get('test_f1', None)
        
        print("MODEL CONFIG:")
        print(json.dumps(model_config, indent=2, default=str))
        
        print(f"\n\nLABEL MAP:")
        print(json.dumps(label_map, indent=2, default=str))
        
        print(f"\n\nTEST F1 SCORE: {test_f1}")
        
        # Now inspect the actual state_dict
        state_dict = checkpoint['model_state_dict']
        print(f"\n\nMODEL STATE DICT STRUCTURE ({len(state_dict)} parameters):")
        
        for i, key in enumerate(list(state_dict.keys())[:20]):  # Show first 20
            value = state_dict[key]
            print(f"  {key:<60s} -> {value.shape}")
        
        if len(state_dict) > 20:
            print(f"  ... and {len(state_dict) - 20} more parameters")
    
    except Exception as e:
        print(f"[ERROR] Failed to extract: {e}")
        import traceback
        traceback.print_exc()

print(f"\n\n{'='*80}")
print("SHARE THE OUTPUT ABOVE AND WE'LL BUILD THE LOADER")
print("="*80)