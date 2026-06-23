from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .feature_extractor import MultiLayerFeatureExtractor
from .utils import ensure_dir, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export feature extractor to ONNX.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.category:
        cfg["data"]["category"] = args.category

    category = cfg["data"]["category"]
    image_size = cfg["data"]["image_size"]
    opset = cfg["export"].get("opset", 17)
    device = resolve_device(cfg["train"].get("device", "auto"))
    out_dir = ensure_dir(Path("artifacts") / category)
    out_path = out_dir / "feature_extractor.onnx"

    model = MultiLayerFeatureExtractor(cfg["model"]["layers"]).to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    dynamic_axes = None
    if cfg["export"].get("dynamic_batch", True):
        dynamic_axes = {"input": {0: "batch"}, "features": {0: "batch"}}

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["features"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Saved ONNX: {out_path}")


if __name__ == "__main__":
    main()
