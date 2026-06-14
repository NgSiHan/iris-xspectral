"""
Build the cross-spectral test CSV for VIS-IrisFormer.

Pools real VIS probes (test split) + synthetic VIS gallery (from IPGAN translation)
into a single CSV. VIS-IrisFormer's _data_loader_equal() auto-generates
genuine/impostor pairs from class labels.

Usage:
  python -m src.iris_xspectral.build_matcher_protocol \
      --test_subjects 141-209
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_subject_range(s):
    parts = s.split("-")
    lo, hi = int(parts[0]), int(parts[1])
    return {f"{i:03d}" for i in range(lo, hi + 1)}


def main():
    parser = argparse.ArgumentParser(
        description="Build cross-spectral matcher protocol CSV"
    )
    parser.add_argument("--dataset", choices=["polyu"], default="polyu")
    parser.add_argument("--processed_dir", type=str, default=None)
    parser.add_argument("--gallery_dir", type=str, default=None,
                        help="Synthetic VIS gallery dir")
    parser.add_argument("--test_subjects", type=str, default="141-209")
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    from src.iris_xspectral import load_paths
    paths = load_paths()

    processed_dir = (
        Path(args.processed_dir)
        if args.processed_dir
        else Path(paths["processed_dir"]) / args.dataset
    )
    gallery_dir = (
        Path(args.gallery_dir)
        if args.gallery_dir
        else Path(paths["processed_dir"]) / "gallery_synthVIS"
    )
    csv_path = (
        Path(args.output_csv)
        if args.output_csv
        else PROJECT_ROOT / "Protocols" / "PolyU" / "xspectral_test.csv"
    )

    subjects = parse_subject_range(args.test_subjects)
    vis_dir = processed_dir / "vis"

    rows = []
    class_map = {}
    next_idx = 0
    n_probe = 0
    n_gallery = 0

    # Real VIS probes (test split)
    if vis_dir.exists():
        for class_dir in sorted(vis_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = class_dir.name
            parts = class_id.split("_")
            if len(parts) != 2 or parts[0] not in subjects:
                continue

            if class_id not in class_map:
                class_map[class_id] = next_idx
                next_idx += 1

            idx = class_map[class_id]
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in (".png", ".jpg", ".bmp"):
                    rel = img.relative_to(PROJECT_ROOT)
                    rows.append(f"{rel.as_posix()},{idx}")
                    n_probe += 1

    # Synthetic VIS gallery (translated NIR→VIS)
    if gallery_dir.exists():
        for class_dir in sorted(gallery_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = class_dir.name
            parts = class_id.split("_")
            if len(parts) != 2 or parts[0] not in subjects:
                continue

            if class_id not in class_map:
                class_map[class_id] = next_idx
                next_idx += 1

            idx = class_map[class_id]
            for img in sorted(class_dir.iterdir()):
                if img.suffix.lower() in (".png", ".jpg", ".bmp"):
                    rel = img.relative_to(PROJECT_ROOT)
                    rows.append(f"{rel.as_posix()},{idx}")
                    n_gallery += 1

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("iris_img_path,class_index\n")
        for row in rows:
            f.write(row + "\n")

    print(f"Cross-spectral protocol CSV: {csv_path}")
    print(f"  Real VIS probes:     {n_probe}")
    print(f"  Synthetic VIS gallery: {n_gallery}")
    print(f"  Total images:        {n_probe + n_gallery}")
    print(f"  Classes:             {len(class_map)}")

    if n_gallery == 0:
        print("\nWARNING: No synthetic gallery found. "
              "Run translate_gallery.py first.")


if __name__ == "__main__":
    main()
