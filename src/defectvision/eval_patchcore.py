from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from .data import make_loader
from .patchcore_lite import PatchCoreLite
from .utils import ensure_dir, load_config, resolve_device, save_json
from .viz import save_prediction_grid, tensor_to_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PatchCore-lite on MVTec test split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--max_examples", type=int, default=24)
    return parser.parse_args()


def safe_metric(fn, y_true, y_score):
    try:
        return float(fn(y_true, y_score))
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.category:
        cfg["data"]["category"] = args.category

    category = cfg["data"]["category"]
    device = resolve_device(cfg["train"].get("device", "auto"))
    image_size = cfg["data"]["image_size"]
    out_dir = ensure_dir(Path("artifacts") / category)
    examples_dir = ensure_dir(out_dir / "examples")

    model = PatchCoreLite.load(out_dir / "patchcore_lite.pt", device=device)
    test_loader = make_loader(
        root=cfg["data"]["root"],
        category=category,
        split="test",
        image_size=image_size,
        batch_size=cfg["inference"].get("batch_size", 8),
        num_workers=cfg["data"].get("num_workers", 2),
    )

    all_labels, all_scores = [], []
    all_pixel_labels, all_pixel_scores = [], []
    saved = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        pred = model.predict_batch(batch["image"], output_size=(image_size, image_size))
        scores = pred["image_scores"].numpy()
        maps = pred["anomaly_maps"].numpy()
        labels = batch["label"].numpy()
        masks = batch["mask"].numpy()[:, 0]

        all_labels.extend(labels.tolist())
        all_scores.extend(scores.tolist())
        all_pixel_labels.append(masks.reshape(-1))
        all_pixel_scores.append(maps.reshape(-1))

        for i in range(len(labels)):
            if saved >= args.max_examples:
                break
            path = Path(batch["path"][i])
            defect_type = batch["defect_type"][i]
            title = f"{path.name} | {defect_type} | score={scores[i]:.4f}"
            save_prediction_grid(
                image=tensor_to_image(batch["raw"][i]),
                anomaly_map=maps[i],
                mask=masks[i] if labels[i] == 1 else None,
                out_path=examples_dir / f"{saved:03d}_{defect_type}_{path.stem}.png",
                title=title,
            )
            saved += 1

    y_true = np.asarray(all_labels)
    y_score = np.asarray(all_scores)
    pixel_true = np.concatenate(all_pixel_labels)
    pixel_score = np.concatenate(all_pixel_scores)

    metrics = {
        "category": category,
        "num_test_images": int(len(y_true)),
        "image_auroc": safe_metric(roc_auc_score, y_true, y_score),
        "image_average_precision": safe_metric(average_precision_score, y_true, y_score),
        "pixel_auroc": safe_metric(roc_auc_score, pixel_true, pixel_score),
        "pixel_average_precision": safe_metric(average_precision_score, pixel_true, pixel_score),
    }
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Saved examples: {examples_dir}")


if __name__ == "__main__":
    main()
