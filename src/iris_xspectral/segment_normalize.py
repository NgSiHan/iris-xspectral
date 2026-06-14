"""
Segment raw iris images and produce 64x512 normalized strips.

VIS:  LightIrisNet (MobileNetV3 DeepLabv3+) -> masks -> circle fit -> Daugman
NIR:  iris-fm-tools (DINOv3 + circlenet) -> circles directly -> Daugman

Both share: to_mono -> circles+mask -> quality_ok -> cart_to_pol -> enhance_strip

Usage:
  python -m src.iris_xspectral.segment_normalize \
      --dataset polyu --spectrum both \
      --output_dir data/processed/polyu \
      --preview 20
"""

import argparse
import math
import sys
from math import pi
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "vendor"))
sys.path.insert(0, str(PROJECT_ROOT / "vendor" / "iris_fm_tools"))

from cvrl_seg.iris_norm import STRIP_H, STRIP_W, enhance_strip, to_mono


# ─── Standalone Daugman rubber-sheet (from CVRLSegmenter._cart_to_pol) ────


def _grid_sample_adjusted(inp, grid, mode):
    N, C, H, W = inp.shape
    gx = grid[:, :, :, 0]
    gy = grid[:, :, :, 1]
    gx = ((gx + 1) / 2 * W - 0.5) / (W - 1) * 2 - 1
    gy = ((gy + 1) / 2 * H - 0.5) / (H - 1) * 2 - 1
    newgrid = torch.stack([gx, gy], dim=-1)
    return torch.nn.functional.grid_sample(
        inp, newgrid, mode=mode, align_corners=True, padding_mode="border"
    )


@torch.inference_mode()
def cart_to_pol(image_np, mask_np, pupil_xyr, iris_xyr, device,
                polar_h=STRIP_H, polar_w=STRIP_W):
    """Daugman rubber-sheet normalization via PyTorch grid_sample.

    Args:
        image_np: uint8 HxW grayscale numpy array.
        mask_np:  uint8 HxW mask (255=iris, 0=background).
        pupil_xyr, iris_xyr: array-like [cx, cy, radius].
        device: torch device.

    Returns:
        (strip uint8 [polar_h, polar_w], mask uint8 [polar_h, polar_w])
    """
    img = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    msk = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    height, width = image_np.shape[:2]

    pupil_t = torch.tensor(pupil_xyr, dtype=torch.float32).unsqueeze(0).to(device)
    iris_t = torch.tensor(iris_xyr, dtype=torch.float32).unsqueeze(0).to(device)

    theta = (2 * pi * torch.linspace(0, polar_w - 1, polar_w) / polar_w).to(device)
    cos_t = torch.cos(theta).reshape(1, polar_w)
    sin_t = torch.sin(theta).reshape(1, polar_w)

    pxc = pupil_t[:, 0].reshape(-1, 1) + pupil_t[:, 2].reshape(-1, 1) @ cos_t
    pyc = pupil_t[:, 1].reshape(-1, 1) + pupil_t[:, 2].reshape(-1, 1) @ sin_t
    ixc = iris_t[:, 0].reshape(-1, 1) + iris_t[:, 2].reshape(-1, 1) @ cos_t
    iyc = iris_t[:, 1].reshape(-1, 1) + iris_t[:, 2].reshape(-1, 1) @ sin_t

    radius = (torch.linspace(1, polar_h, polar_h) / polar_h).reshape(-1, 1).to(device)
    px = torch.matmul((1 - radius), pxc.reshape(-1, 1, polar_w))
    py = torch.matmul((1 - radius), pyc.reshape(-1, 1, polar_w))
    ix = torch.matmul(radius, ixc.reshape(-1, 1, polar_w))
    iy = torch.matmul(radius, iyc.reshape(-1, 1, polar_w))

    x_norm = (((px + ix).float() - 1) / (width - 1)) * 2 - 1
    y_norm = (((py + iy).float() - 1) / (height - 1)) * 2 - 1
    grid = torch.cat([x_norm.unsqueeze(-1), y_norm.unsqueeze(-1)], dim=-1)

    img_polar = torch.clamp(
        torch.round(_grid_sample_adjusted(img, grid, "bilinear")), 0, 255
    )
    mask_polar = (_grid_sample_adjusted(msk, grid, "nearest") > 0.5).long() * 255

    return (
        img_polar[0, 0].cpu().numpy().astype(np.uint8),
        mask_polar[0, 0].cpu().numpy().astype(np.uint8),
    )


# ─── Quality checks (from CVRLSegmenter._quality_ok) ─────────────────────


def quality_ok(pupil_xyr, iris_xyr, mask, min_pupil_r=12, min_iris_r=16):
    """Biological sanity checks. Returns False -> treat as seg failure."""
    px, py, pr = float(pupil_xyr[0]), float(pupil_xyr[1]), float(pupil_xyr[2])
    ix, iy, ir = float(iris_xyr[0]), float(iris_xyr[1]), float(iris_xyr[2])
    if ir <= pr:
        return False
    if pr < min_pupil_r or ir < min_iris_r:
        return False
    alpha = pr / ir
    if not (0.1 <= alpha <= 0.8):
        return False
    if math.hypot(px - ix, py - iy) / ir > 0.5:
        return False
    visible = (mask > 0).sum()
    denom = pi * (ir + pr) * (ir - pr)
    if denom <= 0 or (visible / denom) < 0.1:
        return False
    return True


# ─── Circle fitting helpers ──────────────────────────────────────────────


def circle_from_mask(mask_bin):
    """Fit minimum enclosing circle from largest contour in binary mask.

    Returns (cx, cy, radius) as ndarray, or None.
    """
    m = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    return np.array([cx, cy, r])


def synth_mask(pupil_xyr, iris_xyr, h, w):
    """Create a synthetic iris annulus mask from circle parameters."""
    mask = np.zeros((h, w), dtype=np.uint8)
    ix, iy, ir = int(round(iris_xyr[0])), int(round(iris_xyr[1])), int(round(iris_xyr[2]))
    px, py, pr = int(round(pupil_xyr[0])), int(round(pupil_xyr[1])), int(round(pupil_xyr[2]))
    cv2.circle(mask, (ix, iy), ir, 255, -1)
    cv2.circle(mask, (px, py), pr, 0, -1)
    return mask


# ─── VIS segmenter (LightIrisNet) ────────────────────────────────────────


def load_vis_segmenter(weights_path, device):
    from lightirisnet.models import IrisNetDeepLab

    model = IrisNetDeepLab(
        backbone="mobilenetv3",
        use_ellipse=True,
        extra_decoder_conv=True,
        pupil_refine_depth=1,
    ).to(device)
    model = model.to(memory_format=torch.channels_last)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


@torch.inference_mode()
def segment_vis(model, mono, device):
    """Segment VIS image -> (pupil_xyr, iris_xyr, iris_mask) or Nones."""
    from lightirisnet._test_impl import predict as lin_predict

    img_rgb = np.stack([mono, mono, mono], axis=-1).astype(np.float32) / 255.0

    result, _ = lin_predict(
        model, img_rgb, device=device,
        thr_iris=0.5, thr_pupil=0.5,
        do_post=True, use_ellipse=True,
        amp=(device.type == "cuda"),
        containment_mode="soft",
        inside_thresh=0.85, keep_area_frac=0.90,
    )

    iris_mask = (result["iris"] > 0.5).astype(np.uint8) * 255
    pupil_mask = (result["pupil"] > 0.5).astype(np.uint8) * 255

    pupil_xyr = circle_from_mask(pupil_mask)
    iris_xyr = circle_from_mask(iris_mask)

    if pupil_xyr is None or iris_xyr is None:
        return None, None, None

    return pupil_xyr, iris_xyr, iris_mask


# ─── NIR segmenter (iris-fm-tools DINOv3 + circlenet) ────────────────────


def load_nir_segmenter(dino_repo_dir, dino_weights, circlenet_weights, device):
    from inference import load_dino, load_model

    dino = load_dino(dino_repo_dir, dino_weights, device)
    model, info = load_model(circlenet_weights, device)
    return dino, model, info


@torch.inference_mode()
def segment_nir(dino, model, info, mono, device):
    """Segment NIR image -> (pupil_xyr, iris_xyr, mask) or Nones."""
    from inference import predict_single, scale_to_original

    h, w = mono.shape[:2]
    img_pil = Image.fromarray(np.stack([mono, mono, mono], axis=-1))

    try:
        params = predict_single(img_pil, model, dino, info, device)
    except Exception:
        return None, None, None

    params = scale_to_original(params, info, w, h)

    pupil_xyr = np.array([params[0], params[1], params[2]])
    iris_xyr = np.array([params[3], params[4], params[5]])

    mask = synth_mask(pupil_xyr, iris_xyr, h, w)
    return pupil_xyr, iris_xyr, mask


# ─── PolyU dataset walker ────────────────────────────────────────────────


def walk_polyu(raw_root, spectrum):
    """Yield (path, subject, eye, modality) for PolyU Session 1 images."""
    raw_root = Path(raw_root)
    want = {"VIS", "NIR"} if spectrum == "both" else {spectrum.upper()}

    for subj_dir in sorted(raw_root.iterdir()):
        if not subj_dir.is_dir() or not subj_dir.name.isdigit():
            continue
        subject = subj_dir.name
        for eye_dir in sorted(subj_dir.iterdir()):
            if not eye_dir.is_dir() or eye_dir.name not in ("L", "R"):
                continue
            eye = eye_dir.name
            for spec_dir in sorted(eye_dir.iterdir()):
                if not spec_dir.is_dir() or spec_dir.name not in want:
                    continue
                modality = spec_dir.name
                for img_path in sorted(spec_dir.iterdir()):
                    if img_path.suffix.lower() in (".tiff", ".tif", ".png", ".jpg", ".bmp"):
                        yield img_path, subject, eye, modality


# ─── Preview overlay ─────────────────────────────────────────────────────


def save_preview(mono, pupil_xyr, iris_xyr, mask, strip, out_path):
    h, w = mono.shape
    vis = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    ix, iy, ir = int(iris_xyr[0]), int(iris_xyr[1]), int(iris_xyr[2])
    px, py, pr = int(pupil_xyr[0]), int(pupil_xyr[1]), int(pupil_xyr[2])
    cv2.circle(vis, (ix, iy), ir, (0, 255, 255), 2)
    cv2.circle(vis, (px, py), pr, (255, 255, 0), 2)

    overlay = vis.copy()
    mask_bool = mask > 0
    overlay[mask_bool] = (
        overlay[mask_bool].astype(np.float32) * 0.7
        + np.array([0, 80, 0], dtype=np.float32)
    ).astype(np.uint8)

    strip_vis = cv2.resize(cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR), (w, 64))
    combined = np.vstack([overlay, strip_vis])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


# ─── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Segment + normalize iris images to 64x512 strips"
    )
    parser.add_argument("--dataset", choices=["polyu", "cuviris"], default="polyu")
    parser.add_argument("--spectrum", choices=["vis", "nir", "both"], default="both")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--preview", type=int, default=0,
                        help="Save N preview overlay images to runs/masks/")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_enhance", action="store_true",
                        help="Skip background subtract + CLAHE")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.iris_xspectral import load_paths

    paths = load_paths()
    device = torch.device(args.device)
    do_enhance = not args.no_enhance

    if args.dataset == "polyu":
        raw_root = paths["polyu_session1_raw"]
    else:
        raise NotImplementedError("CUVIRIS support not yet implemented")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(paths["processed_dir"]) / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = PROJECT_ROOT / "runs" / "masks"

    # ── Load models ──────────────────────────────────────────────────────
    weights_dir = Path(paths["weights_dir"])
    want = {"VIS", "NIR"} if args.spectrum == "both" else {args.spectrum.upper()}

    vis_model = None
    nir_dino = nir_model = nir_info = None

    if "VIS" in want:
        print("Loading VIS segmenter (LightIrisNet)...")
        vis_model = load_vis_segmenter(
            str(weights_dir / "best_mobilenetv3.pt"), device
        )

    if "NIR" in want:
        print("Loading NIR segmenter (DINOv3 + circlenet)...")
        nir_dino, nir_model, nir_info = load_nir_segmenter(
            str(PROJECT_ROOT / "vendor" / "iris_fm_tools" / "modules" / "dinov3"),
            str(weights_dir / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
            str(weights_dir / "circlenet.pth"),
            device,
        )

    # ── Process images ───────────────────────────────────────────────────
    counts = {"ok": 0, "seg_fail": 0, "qc_fail": 0, "read_fail": 0}
    preview_count = 0

    images = list(walk_polyu(raw_root, args.spectrum))
    print(f"Found {len(images)} images to process")

    for i, (img_path, subject, eye, modality) in enumerate(images):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i+1}/{len(images)}] {modality} {subject}_{eye} — {img_path.name}")

        try:
            mono = to_mono(img_path, modality)
        except Exception:
            counts["read_fail"] += 1
            continue

        if modality == "VIS":
            pupil_xyr, iris_xyr, mask = segment_vis(vis_model, mono, device)
        else:
            pupil_xyr, iris_xyr, mask = segment_nir(
                nir_dino, nir_model, nir_info, mono, device
            )

        if pupil_xyr is None:
            counts["seg_fail"] += 1
            continue

        if not quality_ok(pupil_xyr, iris_xyr, mask):
            counts["qc_fail"] += 1
            continue

        try:
            strip, polar_mask = cart_to_pol(mono, mask, pupil_xyr, iris_xyr, device)
        except Exception:
            counts["seg_fail"] += 1
            continue

        if do_enhance:
            strip = enhance_strip(strip)

        class_id = f"{subject}_{eye}"
        out_subdir = output_dir / modality.lower() / class_id
        out_subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_subdir / f"{img_path.stem}.png"), strip)

        counts["ok"] += 1

        if args.preview > 0 and preview_count < args.preview:
            save_preview(
                mono, pupil_xyr, iris_xyr, mask, strip,
                preview_dir / f"{modality}_{subject}_{eye}_{img_path.stem}.png",
            )
            preview_count += 1

    # ── Summary ──────────────────────────────────────────────────────────
    total = sum(counts.values())
    print(f"\n{'='*50}")
    print(f"Processed {total} images:")
    print(f"  OK:        {counts['ok']}")
    print(f"  Seg fail:  {counts['seg_fail']}")
    print(f"  QC fail:   {counts['qc_fail']}")
    print(f"  Read fail: {counts['read_fail']}")
    if total > 0:
        print(f"  Success:   {counts['ok'] / total * 100:.1f}%")

    vis_dir = output_dir / "vis"
    nir_dir = output_dir / "nir"
    vis_classes = (
        {d.name for d in vis_dir.iterdir() if d.is_dir()} if vis_dir.exists() else set()
    )
    nir_classes = (
        {d.name for d in nir_dir.iterdir() if d.is_dir()} if nir_dir.exists() else set()
    )
    if vis_classes or nir_classes:
        print(f"\n  VIS classes: {len(vis_classes)}")
        print(f"  NIR classes: {len(nir_classes)}")
        print(f"  Both (paired): {len(vis_classes & nir_classes)}")


if __name__ == "__main__":
    main()
