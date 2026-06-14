"""
Evaluate cross-spectral iris matching: real VIS probes vs synthetic VIS gallery.

Uses VIS-IrisFormer (sota.pth) to compute EER, TAR@FAR, and optionally
save pyeer report + DET curve data.

Usage:
  python -m src.iris_xspectral.evaluate_xspectral --save_report
"""

import argparse
import sys
from argparse import Namespace
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "vis_irisformer"))


def _detect_pos_embed(checkpoint_path):
    """Detect position embedding type from checkpoint keys."""
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    has_rope = any("rope" in k for k in ckpt["mae_model"].keys())
    return "rope2d" if has_rope else "cosine"


def _make_test_args(checkpoint_path, **overrides):
    """Build VIS-IrisFormer test args without calling parse_args() on sys.argv."""
    pos_embed = _detect_pos_embed(checkpoint_path)
    print(f"  Auto-detected position_embedding: {pos_embed}")
    defaults = dict(
        use_gpu=True,
        gpu_ids=[0],
        workers=4,
        sample_pairs_number=10000,
        batch_size=256,
        input_size=(64, 512),
        patch_size=(16, 16),
        mask_ratio=0.0,
        in_feats=256,
        ft_pool="map",
        position_embedding=pos_embed,
        bottleneck=False,
        bottleneck_feats=768,
        save_report=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-spectral iris matching evaluation"
    )
    parser.add_argument("--protocol_csv", type=str, default=None,
                        help="Path to cross-spectral test CSV")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_pairs", type=int, default=10000)
    parser.add_argument("--save_report", action="store_true")
    args = parser.parse_args()

    from src.iris_xspectral import load_paths
    from data_config.xspectral_config import Config
    from test import Tester

    paths = load_paths()
    weights_dir = Path(paths["weights_dir"])
    results_dir = Path(paths["results_dir"])

    csv_path = (
        Path(args.protocol_csv)
        if args.protocol_csv
        else PROJECT_ROOT / "Protocols" / "PolyU" / "xspectral_test.csv"
    )

    if not csv_path.exists():
        print(f"ERROR: Protocol CSV not found: {csv_path}")
        print("Run build_matcher_protocol.py first.")
        sys.exit(1)

    checkpoint_path = weights_dir / "sota.pth"
    save_path = results_dir / "polyu_xspectral"
    save_path.mkdir(parents=True, exist_ok=True)

    test_args = _make_test_args(
        checkpoint_path,
        save_report=args.save_report,
        batch_size=args.batch_size,
        sample_pairs_number=args.sample_pairs,
    )

    config = Config(
        root_path=str(PROJECT_ROOT),
        test_csv=str(csv_path),
        num_class=209,
        data_name="PolyU_XSpectral",
    )

    print("Cross-spectral evaluation")
    print(f"  Protocol: {csv_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Results: {save_path}")

    tester = Tester(test_args, config, str(checkpoint_path), str(save_path))
    tester.test_runner()


if __name__ == "__main__":
    main()
