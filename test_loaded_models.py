# test_loaded_models.py
import torch
from pathlib import Path
from finetune.models import SingleStreamClassifier, FusionClassifier, load_pretrained_backbone

MODEL_DIR = Path("finetune/models/dashboard_models")
device = torch.device("cpu")

def try_load_and_forward(fname, is_fusion=False):
    p = MODEL_DIR / fname
    print("\n==>", p)
    m = torch.load(str(p), map_location=device, weights_only=False)
    print("loaded type:", type(m))
    m.eval()
    with torch.no_grad():
        if is_fusion:
            x1 = torch.randn(1,6,150)
            x2 = torch.randn(1,6,150)
            out = m(x1, x2)
        else:
            x = torch.randn(1,6,150)
            out = m(x)
    print("forward ok, output shape:", getattr(out, "shape", None))

# Try full files first (converter creates *_full.pth)
try:
    try_load_and_forward("phone_classifier_full.pth", is_fusion=False)
except Exception as e:
    print("phone full failed:", e)
try:
    try_load_and_forward("watch_classifier_full.pth", is_fusion=False)
except Exception as e:
    print("watch full failed:", e)
try:
    try_load_and_forward("fusion_classifier_full.pth", is_fusion=True)
except Exception as e:
    print("fusion full failed:", e)

# As fallback, test original filenames if converter didn't create *_full versions
for fname, is_f in [("phone_classifier.pth", False), ("watch_classifier.pth", False), ("fusion_classifier.pth", True)]:
    try:
        try_load_and_forward(fname, is_fusion=is_f)
    except Exception as e:
        print(f"{fname} load/forward failed:", e)
