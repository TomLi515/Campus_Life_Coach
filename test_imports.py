# convert_and_export_models.py
"""
Run from your repository root. This script will:
 - attempt to make `finetune.models` importable by adding likely src paths to sys.path
 - allowlist the finetune model classes and load the pickled full-model .pth files
 - save new: *_classifier_full.pth (full nn.Module), *_classifier_state_dict.pth and *_classifier.pt (TorchScript)
"""
import sys
import traceback
from pathlib import Path

# 1) Add likely "finetune/src" or "src" folders to sys.path so `from finetune.models import ...` works.
root = Path(__file__).resolve().parent

candidates = [
    root / "finetune" / "src",
    root / "src",                    # e.g., repo_root/src/finetune
    root / "finetune",               # sometimes package already in this dir structure
    root.parent / "finetune" / "src",
    root.parent / "src" / "finetune" / "src",
]

added = []
for p in candidates:
    if p.exists() and p.is_dir():
        sys.path.insert(0, str(p))
        added.append(str(p))

print("Added to sys.path (attempt):", added)

# Also print current sys.path first items for debugging
print("sys.path[0:6]:", sys.path[:6])

# Now attempt to import the classes
try:
    from finetune.models import SingleStreamClassifier, FusionClassifier  # noqa
    print("Imported SingleStreamClassifier, FusionClassifier from finetune.models")
except Exception as e:
    print("Failed to import finetune.models classes. Exception:")
    traceback.print_exc()
    print("\nDouble-check where your `finetune` package lives relative to this script.")
    print("If finetune lives in a non-standard place, add its 'src' folder to PYTHONPATH or the `candidates` list above.")
    sys.exit(1)

# Proceed with conversion
try:
    import torch
except Exception as e:
    print("Failed to import torch:", e)
    sys.exit(1)

MODEL_DIR = Path("finetune/models/dashboard_models")
if not MODEL_DIR.exists():
    print("Model directory not found:", MODEL_DIR)
    sys.exit(1)

device = torch.device("cpu")

items = [
    ("phone", MODEL_DIR / "phone_classifier.pth"),
    ("watch", MODEL_DIR / "watch_classifier.pth"),
    ("fusion", MODEL_DIR / "fusion_classifier.pth"),
]

# allowlist objects for safe unpickle
safe_globals = [SingleStreamClassifier, FusionClassifier]

for name, p in items:
    print("\n" + "="*60)
    print(f"Processing: {p}")
    if not p.exists():
        print("SKIP: file does not exist:", p)
        continue
    try:
        # Prefer context manager if available (PyTorch >= 2.6)
        ctx = None
        try:
            ctx = torch.serialization.safe_globals(safe_globals)
        except Exception:
            try:
                torch.serialization.add_safe_globals(safe_globals)
            except Exception:
                pass

        if ctx is not None:
            with ctx:
                loaded = torch.load(str(p), map_location="cpu", weights_only=False)
        else:
            # add_safe_globals may be persistent for the process
            try:
                torch.serialization.add_safe_globals(safe_globals)
            except Exception:
                pass
            loaded = torch.load(str(p), map_location="cpu", weights_only=False)

        print("Loaded type:", type(loaded))

        if isinstance(loaded, torch.nn.Module):
            model = loaded.to(device).eval()
            # Save full module (new file)
            full_path = MODEL_DIR / f"{name}_classifier_full.pth"
            torch.save(model, str(full_path))
            print(f"[saved] Full nn.Module saved -> {full_path}")

            # Backup state_dict
            ckpt_path = MODEL_DIR / f"{name}_classifier_state_dict.pth"
            torch.save({'model_state_dict': model.state_dict()}, str(ckpt_path))
            print(f"[saved] State dict saved -> {ckpt_path}")

            # Try to export TorchScript (trace)
            try:
                if name == "fusion":
                    example = (torch.randn(1, 6, 150), torch.randn(1, 6, 150))
                else:
                    example = torch.randn(1, 6, 150)
                traced = torch.jit.trace(model, example)
                ts_path = MODEL_DIR / f"{name}_classifier.pt"
                traced.save(str(ts_path))
                print(f"[saved] TorchScript exported -> {ts_path}")
            except Exception as e:
                print(f"[warn] TorchScript trace failed for {name}: {e}")
                traceback.print_exc()

        elif isinstance(loaded, dict):
            print("Loaded a dict. Keys:", list(loaded.keys())[:20])
            sd = None
            if "model_state_dict" in loaded:
                sd = loaded["model_state_dict"]
            elif "state_dict" in loaded:
                sd = loaded["state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in loaded.values()):
                sd = loaded
            if sd is not None:
                ckpt_path = MODEL_DIR / f"{name}_classifier_state_dict_only.pth"
                torch.save({'model_state_dict': sd}, str(ckpt_path))
                print(f"[saved] Extracted state_dict -> {ckpt_path}")
                print("To convert state_dict to full model you must instantiate the model class and load it.")
            else:
                # Save raw copy
                torch.save(loaded, MODEL_DIR / f"{name}_raw_loaded.pth")
                print("Saved raw dict to file for inspection.")
        else:
            print("Unhandled type returned by torch.load:", type(loaded))
    except Exception as e:
        print("ERROR while processing", p)
        traceback.print_exc()

print("\nConversion script finished. Check files in:", MODEL_DIR)
print("If *_classifier_full.pth files were created, update your dashboard to load them (or rename them to expected names).")
