from __future__ import annotations

import argparse
from pathlib import Path

from .data import make_loader
from .patchcore_lite import PatchCoreLite
from .utils import ensure_dir, load_config, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PatchCore-lite memory bank.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.category:
        cfg["data"]["category"] = args.category
    set_seed(cfg.get("seed", 42))

    category = cfg["data"]["category"]
    device = resolve_device(cfg["train"].get("device", "auto"))
    print(f"[DefectVision-TRT] category={category} device={device}")

    train_loader = make_loader(
        root=cfg["data"]["root"],
        category=category,
        split="train",
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 2),
    )

    model = PatchCoreLite(
        layers=cfg["model"]["layers"],
        max_memory_patches=cfg["model"]["max_memory_patches"],
        distance_chunk_size=cfg["model"]["distance_chunk_size"],
        score_topk_ratio=cfg["model"]["score_topk_ratio"],
        device=device,
    )
    model.fit(train_loader)

    out_dir = ensure_dir(Path("artifacts") / category)
    out_path = out_dir / "patchcore_lite.pt"
    model.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
