#!/usr/bin/env python3
"""Validate that the iris-xspectral environment is correctly set up.

Checks: Python version, torch/CUDA, weight files, dataset paths, vendor imports.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

errors = 0
warnings = 0


def check(label, condition, note=""):
    global errors, warnings
    if condition:
        print(f"  [{PASS}] {label}")
    else:
        print(f"  [{FAIL}] {label}" + (f"  -- {note}" if note else ""))
        errors += 1


def warn(label, condition, note=""):
    global warnings
    if condition:
        print(f"  [{PASS}] {label}")
    else:
        print(f"  [{WARN}] {label}" + (f"  -- {note}" if note else ""))
        warnings += 1


# --- Python ---
print("\n=== Python ===")
check(f"Python >= 3.8 (have {sys.version.split()[0]})",
      sys.version_info >= (3, 8))

# --- PyTorch ---
print("\n=== PyTorch ===")
try:
    import torch
    check(f"torch importable (v{torch.__version__})", True)
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        check(f"CUDA available ({torch.cuda.get_device_name(0)})", True)
    else:
        warn("CUDA available", False, "CPU only — training will be slow")
except ImportError:
    warn("torch importable", False, "pip install torch (required for ML stages)")

# --- Core dependencies ---
print("\n=== Dependencies ===")
for mod in ["cv2", "numpy", "PIL", "yaml", "sklearn", "pandas", "tqdm", "matplotlib"]:
    try:
        __import__(mod)
        check(f"{mod} importable", True)
    except ImportError:
        warn(f"{mod} importable", False, f"pip install {mod}")

try:
    import pyeer
    check("pyeer importable", True)
except ImportError:
    warn("pyeer importable", False, "pip install pyeer (needed for EER metrics)")

try:
    import einops
    check("einops importable", True)
except ImportError:
    warn("einops importable", False, "pip install einops (needed for VIS-IrisFormer)")

# --- Weight files ---
print("\n=== Weight Files ===")
weights_dir = PROJECT_ROOT / "weights"
required_weights = {
    "sota.pth": "VIS-IrisFormer matcher",
    "best_mobilenetv3.pt": "LightIrisNet VIS segmenter",
    "circlenet.pth": "iris-fm-tools circle detection",
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth": "DINOv3 backbone",
    "nestedsharedatrousresunet-006-0.028214-maskIoU-0.938446.pth": "CVRL mask model",
    "resnet18-027-0.008222-maskIoU-0.967159.pth": "CVRL circle model",
}
optional_weights = {
    "eyelidnet_cubic.pth": "iris-fm-tools eyelid (cubic)",
    "eyelidnet_parabola.pth": "iris-fm-tools eyelid (parabola)",
    "h8net.pth": "iris-fm-tools H8 correction",
    "cornernet_live.pth": "iris-fm-tools corner (live)",
    "cornernet_pmi.pth": "iris-fm-tools corner (PMI)",
}

for fname, desc in required_weights.items():
    path = weights_dir / fname
    exists = path.exists()
    size_mb = f" ({path.stat().st_size / 1e6:.1f} MB)" if exists else ""
    check(f"{fname}{size_mb} — {desc}", exists)

for fname, desc in optional_weights.items():
    path = weights_dir / fname
    exists = path.exists()
    size_mb = f" ({path.stat().st_size / 1e6:.1f} MB)" if exists else ""
    warn(f"{fname}{size_mb} — {desc}", exists)

# --- Dataset paths ---
print("\n=== Dataset Paths ===")
try:
    from src.iris_xspectral import load_paths
    paths = load_paths()
    env = os.environ.get("IRIS_ENV", "windows")
    print(f"  Config: paths.{env}.yaml")

    polyu_raw = paths.get("polyu_session1_raw", "")
    if os.path.isdir(polyu_raw):
        subj_count = len([d for d in os.listdir(polyu_raw) if os.path.isdir(os.path.join(polyu_raw, d))])
        check(f"PolyU raw: {subj_count} subjects at {polyu_raw}", subj_count > 0)
    else:
        warn(f"PolyU raw accessible", False, f"not found: {polyu_raw}")

    polyu_norm = paths.get("polyu_session1_norm", "")
    if os.path.isdir(polyu_norm):
        check(f"PolyU pre-normalized: {polyu_norm}", True)
    else:
        warn("PolyU pre-normalized accessible", False, f"not found: {polyu_norm}")

    cuviris = paths.get("cuviris_root", "")
    if os.path.isdir(cuviris):
        check(f"CUVIRIS: {cuviris}", True)
    else:
        warn("CUVIRIS accessible", False, f"not found: {cuviris} (not needed yet)")

except ImportError:
    warn("Path config loadable", False, "pyyaml not installed — install in ML env")
except Exception as e:
    check(f"Path config loadable", False, str(e))

# --- Vendor imports (require torch — warn-only if torch missing) ---
print("\n=== Vendor Imports ===")
torch_available = "torch" in sys.modules or False
try:
    import torch as _t
    torch_available = True
except ImportError:
    pass

if not torch_available:
    print("  [SKIP] Vendor imports require torch — skipping (install in ML env)")
else:
    # VIS-IrisFormer
    vis_irisformer_dir = PROJECT_ROOT / "vendor" / "vis_irisformer"
    sys.path.insert(0, str(vis_irisformer_dir))
    try:
        from model.Transformers.VIT.mae import MAEVisionTransformers
        check("VIS-IrisFormer model importable", True)
    except Exception as e:
        check("VIS-IrisFormer model importable", False, str(e))

    # LightIrisNet
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "vendor"))
        from lightirisnet.models import IrisNetDeepLab
        check("LightIrisNet model importable", True)
    except Exception as e:
        check("LightIrisNet model importable", False, str(e))

    # CVRL Segmenter
    try:
        from cvrl_seg.cvrl_seg import CVRLSegmenter
        check("CVRL segmenter importable", True)
    except Exception as e:
        check("CVRL segmenter importable", False, str(e))

    # iris-fm-tools (may fail without full dinov3 env — warn only)
    try:
        from iris_fm_tools.inference import load_dino
        check("iris-fm-tools importable", True)
    except Exception as e:
        warn("iris-fm-tools importable", False, str(e))

# --- Summary ---
print(f"\n{'='*52}")
if errors == 0 and warnings == 0:
    print(f"All checks passed!")
elif errors == 0:
    print(f"All critical checks passed. {warnings} warning(s).")
else:
    print(f"{errors} FAILED, {warnings} warning(s). Fix the failures before proceeding.")
sys.exit(1 if errors > 0 else 0)
