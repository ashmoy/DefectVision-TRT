from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch

from .data import imagenet_transform, raw_image_transform
from .patchcore_lite import PatchCoreLite
from .utils import ensure_dir, load_config, resolve_device
from .viz import save_prediction_grid, tensor_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on one image.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.category:
        cfg["data"]["category"] = args.category
    category = cfg["data"]["category"]
    image_size = cfg["data"]["image_size"]
    device = resolve_device(cfg["train"].get("device", "auto"))

    out_dir = ensure_dir(Path("artifacts") / category)
    out_path = Path(args.out) if args.out else out_dir / "single_inference.png"

    model = PatchCoreLite.load(out_dir / "patchcore_lite.pt", device=device)
    image = Image.open(args.image).convert("RGB")
    image_tensor = imagenet_transform(image_size)(image)[None]
    raw_tensor = raw_image_transform(image_size)(image)

    pred = model.predict_batch(image_tensor, output_size=(image_size, image_size))
    score = float(pred["image_scores"][0])
    anomaly_map = pred["anomaly_maps"][0].numpy()

    save_prediction_grid(
        image=tensor_to_image(raw_tensor),
        anomaly_map=anomaly_map,
        out_path=out_path,
        title=f"score={score:.4f}",
    )
    print(f"score={score:.6f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
