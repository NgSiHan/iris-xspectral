"""
Validate VIS-IrisFormer matcher (sota.pth) on VIS-only strips.

1. Builds Protocols/PolyU/vis_intra_test.csv from processed VIS test-split strips.
2. Runs VIS-IrisFormer evaluation: EER, TAR@FAR.
3. Gate: VIS-VIS EER should be < 10%.

Usage:
  python -m src.iris_xspectral.validate_matcher \
      --test_subjects 141-209 \
      --save_report
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


def parse_subject_range(s):
    parts = s.split("-")
    lo, hi = int(parts[0]), int(parts[1])
    return {f"{i:03d}" for i in range(lo, hi + 1)}


def build_vis_csv(processed_dir, subjects, csv_path):
    """Build iris_img_path,class_index CSV for VIS strips."""
    vis_dir = processed_dir / "vis"
    if not vis_dir.exists():
        raise FileNotFoundError(f"No VIS strips at {vis_dir}")

    rows = []
    class_map = {}
    next_idx = 0

    for class_dir in sorted(vis_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        parts = class_id.split("_")
        if len(parts) != 2:
            continue
        subject = parts[0]
        if subject not in subjects:
            continue

        if class_id not in class_map:
            class_map[class_id] = next_idx
            next_idx += 1

        idx = class_map[class_id]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".bmp", ".tiff"):
                rel = img_path.relative_to(PROJECT_ROOT)
                rows.append(f"{rel.as_posix()},{idx}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("iris_img_path,class_index\n")
        for row in rows:
            f.write(row + "\n")

    return len(rows), len(class_map)


def run_evaluation(csv_path, checkpoint_path, save_path, args):
    """Run VIS-IrisFormer Tester with the generated CSV."""
    from data_config.xspectral_config import Config
    from test import Tester

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
        data_name="PolyU_VIS_Intra",
    )

    tester = Tester(test_args, config, str(checkpoint_path), str(save_path))
    tester.test_runner()


def main():
    parser = argparse.ArgumentParser(
        description="Validate VIS-IrisFormer on VIS-only strips"
    )
    parser.add_argument("--dataset", choices=["polyu"], default="polyu")
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--test_subjects", type=str, default="141-209")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sample_pairs", type=int, default=10000)
    parser.add_argument("--save_report", action="store_true")
    args = parser.parse_args()

    from src.iris_xspectral import load_paths
    paths = load_paths()

    processed_dir = (
        Path(args.processed_dir)
        if args.processed_dir
        else Path(paths["processed_dir"]) / args.dataset
    )
    weights_dir = Path(paths["weights_dir"])
    results_dir = Path(paths["results_dir"])

    subjects = parse_subject_range(args.test_subjects)
    csv_path = PROJECT_ROOT / "Protocols" / "PolyU" / "vis_intra_test.csv"

    # Step 1: Build CSV
    print("Building VIS intra-spectral test CSV...")
    n_images, n_classes = build_vis_csv(processed_dir, subjects, csv_path)
    print(f"  {n_images} images, {n_classes} classes -> {csv_path}")

    if n_images == 0:
        print("ERROR: No VIS strips found. Run segment_normalize first.")
        sys.exit(1)

    # Step 2: Run evaluation
    checkpoint_path = weights_dir / "sota.pth"
    save_path = results_dir / "polyu_vis_intra"
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning VIS-IrisFormer evaluation...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Results:    {save_path}")
    run_evaluation(csv_path, checkpoint_path, save_path, args)


if __name__ == "__main__":
    main()
