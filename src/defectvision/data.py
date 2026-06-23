from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label: int
    defect_type: str
    mask_path: Optional[Path] = None


def imagenet_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def raw_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
        ]
    )


def mask_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )


class MVTecDataset(Dataset):
    def __init__(self, root: str | Path, category: str, split: str, image_size: int = 256) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.category_dir = self.root / category
        if not self.category_dir.exists():
            raise FileNotFoundError(f"Category directory not found: {self.category_dir}")

        self.image_tf = imagenet_transform(image_size)
        self.raw_tf = raw_image_transform(image_size)
        self.mask_tf = mask_transform(image_size)
        self.samples = self._scan()
        if not self.samples:
            raise RuntimeError(f"No samples found for category={category}, split={split} in {self.category_dir}")

    def _scan_images(self, folder: Path) -> List[Path]:
        if not folder.exists():
            return []
        return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

    def _mask_for(self, defect_type: str, image_path: Path) -> Optional[Path]:
        if defect_type == "good":
            return None
        gt_dir = self.category_dir / "ground_truth" / defect_type
        if not gt_dir.exists():
            return None
        stem = image_path.stem
        candidates = [
            gt_dir / f"{stem}_mask.png",
            gt_dir / f"{stem}.png",
            gt_dir / f"{stem}_mask.bmp",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = sorted(gt_dir.glob(f"{stem}*"))
        return matches[0] if matches else None

    def _scan(self) -> List[Sample]:
        samples: List[Sample] = []
        if self.split == "train":
            folder = self.category_dir / "train" / "good"
            for path in self._scan_images(folder):
                samples.append(Sample(path, label=0, defect_type="good"))
            return samples

        if self.split != "test":
            raise ValueError("split must be 'train' or 'test'")

        test_root = self.category_dir / "test"
        for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
            defect_type = defect_dir.name
            label = 0 if defect_type == "good" else 1
            for path in self._scan_images(defect_dir):
                samples.append(Sample(path, label=label, defect_type=defect_type, mask_path=self._mask_for(defect_type, path)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        tensor = self.image_tf(image)
        raw = self.raw_tf(image)
        if sample.mask_path is not None:
            mask_img = Image.open(sample.mask_path).convert("L")
            mask = self.mask_tf(mask_img)
            mask = (mask > 0.5).float()
        else:
            mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        return {
            "image": tensor,
            "raw": raw,
            "mask": mask,
            "label": torch.tensor(sample.label, dtype=torch.long),
            "path": str(sample.image_path),
            "defect_type": sample.defect_type,
        }


def make_loader(root: str | Path, category: str, split: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = MVTecDataset(root=root, category=category, split=split, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers, pin_memory=torch.cuda.is_available())
