
"""Four real photos -> LGM 3D Gaussian Splatting PLY.

Place this file in the root of https://github.com/3DTopia/LGM.
This does not use ImageDream or change LGM model weights.

Example (PowerShell):
  python infer_multiview.py --images data/my_object/0.jpg data/my_object/90.jpg data/my_object/180.jpg data/my_object/270.jpg --reverse-order

For testing preprocessing without CUDA/LGM:
  python infer_multiview.py --images data/my_object/0.jpg data/my_object/90.jpg data/my_object/180.jpg data/my_object/270.jpg --preprocess-only
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an LGM Gaussian Splatting PLY using four real photographs."
    )
    parser.add_argument(
        "--images", nargs=4, type=Path, required=True,
        metavar=("VIEW_0", "VIEW_90", "VIEW_180", "VIEW_270"),
        help="Four photographs in turntable rotation order (0, 90, 180, 270 degrees).",
    )
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("pretrained/model_fp16_fixrot.safetensors"),
        help="Official big LGM pretrained checkpoint.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("workspace_multiview/object.ply"),
    )
    parser.add_argument(
        "--no-remove-bg", action="store_true",
        help="Use existing transparent PNG alpha, or assume images already have clean white backgrounds.",
    )
    parser.add_argument(
        "--reverse-order", action="store_true",
        help="Map the last view to LGM's +90-degree camera slot (reverse turntable direction).",
    )
    parser.add_argument(
        "--elevation", type=float, default=0.0,
        help="LGM camera elevation in degrees. Default 0 (horizontal views).",
    )
    parser.add_argument(
        "--preprocess-only", action="store_true",
        help="Only save preprocessed images; do not load GPU model/checkpoint.",
    )
    args = parser.parse_args()

    if not (-90 < args.elevation < 90):
        parser.error("--elevation must be between -90 and 90 degrees (exclusive).")
    for path in args.images:
        if not path.is_file():
            parser.error(f"Image not found: {path}")
    if args.output.suffix.lower() != ".ply":
        parser.error("--output must end in .ply")
    return args


def load_rgba(path: Path, remove_bg: bool, session_holder: dict) -> Image.Image:
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source)
        img = source.convert("RGBA")

    # Already-masked transparent PNGs are preferred (no extra segmentation errors).
    alpha_extrema = img.getchannel("A").getextrema()
    if alpha_extrema is not None and alpha_extrema[0] < 255:
        return img

    if not remove_bg:
        return img

    try:
        import rembg
    except ImportError as exc:
        raise RuntimeError(
            "Background removal needs rembg. Install rembg and onnxruntime, "
            "or provide transparent PNGs / use --no-remove-bg."
        ) from exc

    if session_holder.get("session") is None:
        print("[INFO] Loading rembg background-removal model (first run may download weights)...")
        session_holder["session"] = rembg.new_session()

    rgb = np.asarray(img.convert("RGB"))
    result = rembg.remove(rgb, session=session_holder["session"])
    return Image.fromarray(np.asarray(result, dtype=np.uint8)).convert("RGBA")


def crop_and_center(
    rgba: Image.Image,
    target_size: int = 256,
    border_ratio: float = 0.2
) -> Image.Image:
    """Fit alpha-bounded foreground in square, with white background.

    Each view is processed separately, analogous to the official image preprocessing.
    This is a first-pass baseline; pose/scale consistency is an experiment to assess.
    """
    rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.where(alpha > 8)
    if xs.size == 0:
        raise ValueError("Foreground mask is empty. Check alpha mask / background removal.")

    bbox = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1
    )
    foreground = rgba.crop(bbox)
    foreground_w, foreground_h = foreground.size
    max_occupied_side = int(round(target_size * (1.0 - 2.0 * border_ratio)))
    if max_occupied_side <= 0:
        raise ValueError("Invalid border_ratio")

    scale = min(
        max_occupied_side / foreground_w,
        max_occupied_side / foreground_h
    )
    new_wh = (
        max(1, int(round(foreground_w * scale))),
        max(1, int(round(foreground_h * scale))),
    )
    foreground = foreground.resize(new_wh, Image.Resampling.LANCZOS)
    centered = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 0))
    x = (target_size - new_wh[0]) // 2
    y = (target_size - new_wh[1]) // 2
    centered.paste(foreground, (x, y))
    result = Image.new("RGBA", centered.size, (255, 255, 255, 255))
    result.alpha_composite(centered)
    return result.convert("RGB")


def save_inputs(args: argparse.Namespace, size: int = 256) -> np.ndarray:
    """Return preprocessed RGB images in the actual order sent to LGM."""

    result_dir = args.output.parent
    processed_dir = result_dir / "processed_inputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    order = [0, 3, 2, 1] if args.reverse_order else [0, 1, 2, 3]
    session_holder = {}
    images = []

    for slot, source_index in enumerate(order):
        path = args.images[source_index]
        rgba = load_rgba(path, not args.no_remove_bg, session_holder)
        img = crop_and_center(rgba, target_size=size)
        img.save(processed_dir / f"slot_{slot}_source_{source_index}.png")
        images.append(np.asarray(img, dtype=np.uint8))
        print(f"[INFO] LGM view {slot}: {path}")

    overview = Image.new("RGB", (size * 2, size * 2), (255, 255, 255))
    for i, img in enumerate(images):
        overview.paste(
            Image.fromarray(img),
            ((i % 2) * size, (i // 2) * size)
        )
    overview.save(result_dir / "input_grid.png")

    manifest = {
        "source_images_in_turntable_order": [str(p) for p in args.images],
        "lgm_slot_source_indices": order,
        "reverse_order": bool(args.reverse_order),
        "elevation_degrees": args.elevation,
        "background_removed_if_needed": not args.no_remove_bg,
        "lgm_input_size": size,
        "note": "LGM camera slots are 0, 90, 180, 270 degrees. View mapping must be verified on real photos.",
    }
    (result_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    return np.stack(images, axis=0)


def infer(args: argparse.Namespace, image_uint8: np.ndarray) -> None:
    # Import the heavy packages only after preprocessing, so --preprocess-only works without GPU.
    import torch
    import torch.nn.functional as F
    import torchvision.transforms.functional as TF
    from safetensors.torch import load_file
    from core.options import config_defaults
    from core.models import LGM

    if not torch.cuda.is_available():
        raise RuntimeError("LGM's Gaussian renderer requires CUDA. CUDA GPU not found.")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Download the official model_fp16_fixrot.safetensors checkpoint first."
        )

    device = torch.device("cuda")
    opt = copy.deepcopy(config_defaults["big"])

    opt.lambda_lpips = 0.0  # inference does not need the LPIPS training loss network
    model = LGM(opt)

    if args.checkpoint.suffix.lower() == ".safetensors":
        state = load_file(str(args.checkpoint), device="cpu")
    else:
        # Only load PyTorch checkpoints you trust.
        state = torch.load(
            str(args.checkpoint),
            map_location="cpu",
            weights_only=True
        )

    loaded = model.load_state_dict(state, strict=False)
    if loaded.missing_keys or loaded.unexpected_keys:
        print(
            f"[WARN] Missing keys: {len(loaded.missing_keys)}, "
            f"unexpected keys: {len(loaded.unexpected_keys)}"
        )
        print("[WARN] Confirm your checkpoint matches the official 'big' model.")
        if len(loaded.missing_keys) > 20 or len(loaded.unexpected_keys) > 20:
            raise RuntimeError("Checkpoint appears incompatible with the big LGM model.")

    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] LGM input size: {opt.input_size}")
    print(f"[INFO] LGM splat size: {opt.splat_size}")

    model = model.half().to(device).eval()
    try:
        rays = model.prepare_default_rays(
            device,
            elevation=args.elevation
        )

        # image_uint8 is preprocessed RGB; resize again for safety to the configured model input size.
        image = torch.from_numpy(image_uint8.copy()).permute(
            0, 3, 1, 2
        ).to(device).float() / 255.0

        image = F.interpolate(
            image,
            size=(opt.input_size, opt.input_size),
            mode="bilinear",
            align_corners=False
        )
        image = TF.normalize(
            image,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )

        inputs = torch.cat([image, rays], dim=1).unsqueeze(0)
        # [1, 4, 9, H, W]
        print(f"[INFO] Inference input: {tuple(inputs.shape)}")

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                gaussians = model.forward_gaussians(inputs)

            args.output.parent.mkdir(parents=True, exist_ok=True)
            model.gs.save_ply(gaussians, str(args.output))

        print(f"[SUCCESS] Saved: {args.output}")
    finally:
        del model
        torch.cuda.empty_cache()


def main() -> None:
    args = get_args()
    images = save_inputs(args)
    print(
        f"[INFO] Saved input preprocessing preview: "
        f"{args.output.parent / 'input_grid.png'}"
    )

    if args.preprocess_only:
        print("[SUCCESS] Preprocessing complete (no model loaded).")
        return

    infer(args, images)


if __name__ == "__main__":
    main()
