from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


def normalize_map(anomaly_map: np.ndarray) -> np.ndarray:
    amin = float(np.min(anomaly_map))
    amax = float(np.max(anomaly_map))
    if amax - amin < 1e-8:
        return np.zeros_like(anomaly_map, dtype=np.float32)
    return ((anomaly_map - amin) / (amax - amin)).astype(np.float32)


def tensor_to_image(raw: torch.Tensor) -> np.ndarray:
    # raw: [3, H, W], range [0, 1]
    arr = raw.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr, 0, 1)


def overlay_heatmap(image: np.ndarray, anomaly_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    norm = normalize_map(anomaly_map)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = (1 - alpha) * image + alpha * heat
    return np.clip(overlay, 0, 1)


def save_prediction_grid(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    out_path: str | Path,
    mask: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = 4 if mask is not None else 3
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 3:
        axes = np.asarray(axes)
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[0].axis("off")

    axes[1].imshow(normalize_map(anomaly_map), cmap="magma")
    axes[1].set_title("anomaly map")
    axes[1].axis("off")

    axes[2].imshow(overlay_heatmap(image, anomaly_map))
    axes[2].set_title("overlay")
    axes[2].axis("off")

    if mask is not None:
        axes[3].imshow(mask, cmap="gray")
        axes[3].set_title("ground truth")
        axes[3].axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
