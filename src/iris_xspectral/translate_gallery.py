"""
Translate NIR test-split strips to synthetic VIS using trained IPGAN generator.

Loads G2 (NIR->VIS) from the IPGAN checkpoint, runs it over NIR strips
from the test split, and saves the synthetic VIS outputs.

Usage:
  python -m src.iris_xspectral.translate_gallery \
      --checkpoint checkpoint/model_polyu_normalized.pt \
      --nir_dir Dataset/NIR_Valid \
      --output_dir data/processed/gallery_synthVIS
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from model import UNet


def main():
    parser = argparse.ArgumentParser(
        description="Translate NIR strips to synthetic VIS via IPGAN G2"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to IPGAN .pt checkpoint")
    parser.add_argument("--nir_dir", type=str, default=None,
                        help="NIR test strips dir (ImageFolder layout)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for synthetic VIS strips")
    parser.add_argument("--feat_dim", type=int, default=128,
                        help="UNet feat_dim (128 for normalized, 256 for cropped)")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from src.iris_xspectral import load_paths
    paths = load_paths()

    device = torch.device(args.device)

    nir_dir = Path(args.nir_dir) if args.nir_dir else Path(paths["ipgan_dataset_dir"]) / "NIR_Valid"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(paths["processed_dir"]) / "gallery_synthVIS"
    )

    # Load G2 (NIR → VIS)
    print(f"Loading IPGAN checkpoint: {args.checkpoint}")
    net = UNet(feat_dim=args.feat_dim)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net.load_state_dict(state["net_2"])
    net.to(device).eval()

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    inv_normalize = transforms.Normalize(
        mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
        std=[1 / 0.5, 1 / 0.5, 1 / 0.5],
    )

    # Walk NIR_Valid/{class_id}/{images}
    total = 0
    classes = 0

    for class_dir in sorted(nir_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_id = class_dir.name
        out_class_dir = output_dir / class_id
        out_class_dir.mkdir(parents=True, exist_ok=True)
        classes += 1

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in (".png", ".jpg", ".bmp", ".tiff"):
                continue

            img_pil = Image.open(img_path).convert("RGB")
            img_tensor = normalize(img_pil).unsqueeze(0).to(device)

            with torch.inference_mode():
                fake_vis, _ = net(img_tensor)

            fake_vis = inv_normalize(fake_vis.squeeze(0).cpu())
            fake_vis = fake_vis.clamp(0, 1)

            # Convert to grayscale (mean of 3 channels)
            gray = fake_vis.mean(dim=0).numpy()
            gray = (gray * 255).astype(np.uint8)

            cv2.imwrite(str(out_class_dir / img_path.name), gray)
            total += 1

        if classes % 20 == 0:
            print(f"  Processed {classes} classes, {total} images...")

    print(f"\nTranslation complete: {total} synthetic VIS strips in {classes} classes")
    print(f"Output: {output_dir}")

    # Quick sanity check
    sample = next(output_dir.rglob("*.png"), None)
    if sample is not None:
        img = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            print(f"Sample: {sample.name}  shape={img.shape}  "
                  f"min={img.min()} max={img.max()} mean={img.mean():.1f}")


if __name__ == "__main__":
    main()
