# SOLUTION 1: Use this if diagnostic says "Full model object"
# Replace the model loading section with this:
import torch 
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_DIR = Path(r"C:\Users\RAJ DAVE\Desktop\thesis\code\new_dataset_folder_cloned\Campus_coach_MLS\final_repo\Campus_Life_Coach\finetune\models\dashboard_models")

def is_model_loaded(m):
    return isinstance(m, torch.nn.Module)

# SIMPLE LOADER for full models
def load_model(model_path, device):
    """Load a full PyTorch model from disk."""
    try:
        if not model_path.exists():
            print(f"[WARNING] Model file not found: {model_path}")
            return None
        
        print(f"[loading] Loading model from: {model_path}")
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.to(device)
        model.eval()
        print(f"[success] Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load all models
print("\n[startup] Loading models...")
phone_model = load_model(MODEL_DIR / "phone_only_classifier.pth", device)
watch_model = load_model(MODEL_DIR / "watch_only_classifier.pth", device)
fusion_model = load_model(MODEL_DIR / "fusion_classifier.pth", device)

print(f"\n[model status]")
print(f"  Phone model loaded: {is_model_loaded(phone_model)}")
print(f"  Watch model loaded: {is_model_loaded(watch_model)}")
print(f"  Fusion model loaded: {is_model_loaded(fusion_model)}\n")