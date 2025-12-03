# convert_and_export_models.py
import torch
from pathlib import Path
import traceback
import sys

# IMPORTANT: import your model classes so we can allowlist them
# Adjust import path if your package layout differs
try:
    from finetune.models import SingleStreamClassifier, FusionClassifier  # noqa
    print("Imported finetune.models classes.")
except Exception as e:
    print("Failed to import finetune.models:", e)
    traceback.print_exc()
    sys.exit(1)

MODEL_DIR = Path("finetune/models/dashboard_models")
out_dir = MODEL_DIR  # overwrite or change to different path
out_dir.mkdir(parents=True, exist_ok=True)

# Files to process (base names)
items = [
    ("phone", MODEL_DIR / "phone_classifier.pth"),
    ("watch", MODEL_DIR / "watch_classifier.pth"),
    ("fusion", MODEL_DIR / "fusion_classifier.pth"),
]

device = torch.device("cpu")

# List classes/functions to allowlist when unpickling
# Add any additional classes referenced by your saved objects if needed
safe_globals = [SingleStreamClassifier, FusionClassifier]

for name, p in items:
    print("\n" + "="*60)
    print(f"Processing {p}")
    if not p.exists():
        print("SKIP: file not found:", p)
        continue
    try:
        # Option A: use context manager to temporarily allow these globals
        # This uses torch.serialization.safe_globals (available in PyTorch 2.6+)
        try:
            ctx = torch.serialization.safe_globals(safe_globals)
        except AttributeError:
            # Fallback to add_safe_globals (older/newer versions may differ)
            try:
                torch.serialization.add_safe_globals(safe_globals)
                ctx = None
            except Exception:
                ctx = None

        if ctx is not None:
            with ctx:
                loaded = torch.load(str(p), map_location="cpu", weights_only=False)
        else:
            # If context manager not available, call add_safe_globals then load
            # Note: add_safe_globals is persistent until process exit
            try:
                torch.serialization.add_safe_globals(safe_globals)
            except Exception:
                pass
            loaded = torch.load(str(p), map_location="cpu", weights_only=False)

        print("torch.load returned object of type:", type(loaded))

        # If it is an nn.Module, good — re-save cleaned copy and optionally export TorchScript
        if isinstance(loaded, torch.nn.Module):
            model = loaded.to(device).eval()
            # Save an explicit full-module file that should be loadable without the allowlist step
            full_path = out_dir / f"{name}_classifier_full.pth"
            torch.save(model, str(full_path))
            print(f"[saved] Full nn.Module saved to: {full_path}")

            # Also save standard checkpoint (state_dict) for backup (optional)
            ckpt_path = out_dir / f"{name}_classifier_state_dict.pth"
            torch.save({'model_state_dict': model.state_dict()}, str(ckpt_path))
            print(f"[saved] State dict saved to: {ckpt_path}")

            # Try to export TorchScript (trace). This is optional but recommended for deploy.
            try:
                # example input shape -- matches your model training (1,6,150)
                if name == "fusion":
                    example = (torch.randn(1, 6, 150), torch.randn(1, 6, 150))
                    traced = torch.jit.trace(model, example)
                else:
                    example = torch.randn(1, 6, 150)
                    traced = torch.jit.trace(model, example)
                ts_path = out_dir / f"{name}_classifier.pt"
                traced.save(str(ts_path))
                print(f"[saved] TorchScript model saved to: {ts_path}")
            except Exception as e:
                print(f"[warn] TorchScript export failed for {name}: {e}")
                traceback.print_exc()

        elif isinstance(loaded, dict):
            # If it's a dict, it might be a checkpoint that contains model_state_dict or state_dict
            keys = list(loaded.keys())
            print("[info] Loaded dict keys:", keys[:20])
            sd = None
            if "model_state_dict" in loaded:
                sd = loaded["model_state_dict"]
            elif "state_dict" in loaded:
                sd = loaded["state_dict"]
            elif all(isinstance(v, torch.Tensor) for v in loaded.values()):
                sd = loaded
            if sd is not None:
                # We cannot instantiate a model here unless we know the exact class & args.
                # But we save the dict to *_state_dict.pth for later usage.
                ckpt_path = out_dir / f"{name}_classifier_state_dict_only.pth"
                torch.save({'model_state_dict': sd}, str(ckpt_path))
                print(f"[saved] Extracted state_dict saved to: {ckpt_path}")
                print("To convert this to a full model you must instantiate the architecture in Python and call load_state_dict.")
            else:
                print("Dict format unrecognized; saved a copy for inspection.")
                torch.save(loaded, out_dir / f"{name}_raw_loaded.pth")
        else:
            print("torch.load returned unexpected type:", type(loaded))
    except Exception as e:
        print("ERROR loading file:", p)
        print(e)
        traceback.print_exc()

print("\nDone. If full models were saved as *_full.pth and/or .pt, copy those filenames into your dashboard MODEL_DIR or update your dashboard to point to them.")
