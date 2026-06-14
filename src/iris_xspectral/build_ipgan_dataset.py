"""
Build IPGAN-compatible Dataset/ layout from processed strips.

Reads from: data/processed/{dataset}/{vis,nir}/{subj_eye}/*.png
Writes to:  Dataset/{VIS,NIR}/{subj_eye}/*.png          (train)
            Dataset/{VIS_Valid,NIR_Valid}/{subj_eye}/*.png (test)

Subject-level split (PolyU):
  Train: subjects 001-140  (~280 classes counting L/R eyes)
  Test:  subjects 141-209  (~138 classes)

Usage:
  python -m src.iris_xspectral.build_ipgan_dataset \
      --dataset polyu --train_subjects 1-140 --test_subjects 141-209
"""

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_subject_range(s):
    """Parse '1-140' into set of zero-padded subject strings like {'001','002',...}."""
    parts = s.split("-")
    lo, hi = int(parts[0]), int(parts[1])
    return {f"{i:03d}" for i in range(lo, hi + 1)}


def scan_processed(processed_dir):
    """Scan data/processed/{vis,nir}/{subj_eye}/ and return per-class info.

    Returns:
        dict[class_id] -> { 'subject': str, 'eye': str,
                            'vis': [Path, ...], 'nir': [Path, ...] }
    """
    classes = defaultdict(lambda: {"vis": [], "nir": []})

    for spec in ("vis", "nir"):
        spec_dir = processed_dir / spec
        if not spec_dir.exists():
            continue
        for class_dir in sorted(spec_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_id = class_dir.name
            parts = class_id.split("_")
            if len(parts) != 2:
                continue
            subject, eye = parts[0], parts[1]
            images = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".bmp", ".tiff")
            )
            classes[class_id][spec] = images
            classes[class_id]["subject"] = subject
            classes[class_id]["eye"] = eye

    return dict(classes)


def main():
    parser = argparse.ArgumentParser(
        description="Build IPGAN Dataset/ from processed strips"
    )
    parser.add_argument("--dataset", choices=["polyu", "cuviris"], default="polyu")
    parser.add_argument("--processed_dir", type=str, default=None,
                        help="Override processed strips dir")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override Dataset output dir")
    parser.add_argument("--train_subjects", type=str, default="1-140")
    parser.add_argument("--test_subjects", type=str, default="141-209")
    parser.add_argument("--min_images", type=int, default=1,
                        help="Minimum images per class per spectrum to include")
    parser.add_argument("--copy", action="store_true", default=True,
                        help="Copy files (default)")
    parser.add_argument("--symlink", action="store_true",
                        help="Symlink instead of copy (Linux)")
    args = parser.parse_args()

    from src.iris_xspectral import load_paths
    paths = load_paths()

    processed_dir = (
        Path(args.processed_dir)
        if args.processed_dir
        else Path(paths["processed_dir"]) / args.dataset
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(paths["ipgan_dataset_dir"])
    )

    train_subjects = parse_subject_range(args.train_subjects)
    test_subjects = parse_subject_range(args.test_subjects)

    print(f"Processed dir: {processed_dir}")
    print(f"Output dir:    {output_dir}")
    print(f"Train subjects: {len(train_subjects)}  Test subjects: {len(test_subjects)}")

    classes = scan_processed(processed_dir)
    print(f"Found {len(classes)} classes in processed dir")

    # Filter: both VIS and NIR must exist with enough images
    dropped = []
    kept = {}
    for class_id, info in sorted(classes.items()):
        vis_n = len(info["vis"])
        nir_n = len(info["nir"])
        if vis_n < args.min_images or nir_n < args.min_images:
            dropped.append({
                "class": class_id,
                "reason": f"insufficient images (VIS={vis_n}, NIR={nir_n})",
            })
            continue
        kept[class_id] = info

    print(f"Kept {len(kept)} paired classes, dropped {len(dropped)}")

    # Split into train / test
    train_classes = {}
    test_classes = {}
    skipped_subjects = []

    for class_id, info in kept.items():
        subj = info["subject"]
        if subj in train_subjects:
            train_classes[class_id] = info
        elif subj in test_subjects:
            test_classes[class_id] = info
        else:
            skipped_subjects.append(class_id)

    print(f"Train: {len(train_classes)} classes  Test: {len(test_classes)} classes")
    if skipped_subjects:
        print(f"  Skipped (not in any split): {skipped_subjects}")

    # Create output directories and copy/link files
    folder_map = {
        "train": {"vis": "VIS", "nir": "NIR"},
        "test": {"vis": "VIS_Valid", "nir": "NIR_Valid"},
    }

    use_link = args.symlink
    total_copied = 0
    per_split_counts = {"train": {"vis": 0, "nir": 0}, "test": {"vis": 0, "nir": 0}}

    for split_name, class_dict in [("train", train_classes), ("test", test_classes)]:
        for class_id, info in sorted(class_dict.items()):
            for spec in ("vis", "nir"):
                src_images = info[spec]
                dst_folder = output_dir / folder_map[split_name][spec] / class_id
                dst_folder.mkdir(parents=True, exist_ok=True)

                for src_path in src_images:
                    dst_path = dst_folder / src_path.name
                    if dst_path.exists():
                        dst_path.unlink()
                    if use_link:
                        dst_path.symlink_to(src_path.resolve())
                    else:
                        shutil.copy2(src_path, dst_path)
                    total_copied += 1
                    per_split_counts[split_name][spec] += 1

    # Write manifest
    manifest = {
        "dataset": args.dataset,
        "train_subjects": args.train_subjects,
        "test_subjects": args.test_subjects,
        "min_images": args.min_images,
        "train_classes": len(train_classes),
        "test_classes": len(test_classes),
        "dropped": dropped,
        "per_split": {
            "train": {
                "classes": len(train_classes),
                "vis_images": per_split_counts["train"]["vis"],
                "nir_images": per_split_counts["train"]["nir"],
            },
            "test": {
                "classes": len(test_classes),
                "vis_images": per_split_counts["test"]["vis"],
                "nir_images": per_split_counts["test"]["nir"],
            },
        },
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Total files written: {total_copied}")
    print(f"  Train: {per_split_counts['train']['vis']} VIS, "
          f"{per_split_counts['train']['nir']} NIR "
          f"({len(train_classes)} classes)")
    print(f"  Test:  {per_split_counts['test']['vis']} VIS, "
          f"{per_split_counts['test']['nir']} NIR "
          f"({len(test_classes)} classes)")
    print(f"Manifest: {manifest_path}")

    if dropped:
        print(f"\nDropped {len(dropped)} classes:")
        for d in dropped[:10]:
            print(f"  {d['class']}: {d['reason']}")
        if len(dropped) > 10:
            print(f"  ... and {len(dropped) - 10} more")


if __name__ == "__main__":
    main()
