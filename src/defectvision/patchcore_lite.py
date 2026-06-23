from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .feature_extractor import MultiLayerFeatureExtractor


class PatchCoreLite:
    def __init__(
        self,
        layers: Iterable[str] = ("layer2", "layer3"),
        max_memory_patches: int = 50_000,
        distance_chunk_size: int = 4096,
        score_topk_ratio: float = 0.01,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.layers = tuple(layers)
        self.max_memory_patches = int(max_memory_patches)
        self.distance_chunk_size = int(distance_chunk_size)
        self.score_topk_ratio = float(score_topk_ratio)
        self.feature_extractor = MultiLayerFeatureExtractor(self.layers).to(self.device).eval()
        self.memory_bank: torch.Tensor | None = None

    @torch.no_grad()
    def _features_to_patches(self, features: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B*H*W, C]
        patches = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        patches = F.normalize(patches, p=2, dim=1)
        return patches

    @torch.no_grad()
    def fit(self, train_loader) -> None:
        banks = []
        for batch in tqdm(train_loader, desc="Building memory bank"):
            images = batch["image"].to(self.device, non_blocking=True)
            features = self.feature_extractor(images)
            patches = self._features_to_patches(features).cpu()
            banks.append(patches)

        memory = torch.cat(banks, dim=0)
        if memory.shape[0] > self.max_memory_patches:
            generator = torch.Generator().manual_seed(42)
            indices = torch.randperm(memory.shape[0], generator=generator)[: self.max_memory_patches]
            memory = memory[indices]
        self.memory_bank = memory.contiguous()

    @torch.no_grad()
    def _min_distances(self, patches: torch.Tensor) -> torch.Tensor:
        if self.memory_bank is None:
            raise RuntimeError("Memory bank is empty. Call fit() or load() first.")
        memory = self.memory_bank.to(patches.device)
        mins = []
        for start in range(0, patches.shape[0], self.distance_chunk_size):
            chunk = patches[start : start + self.distance_chunk_size]
            dist = torch.cdist(chunk, memory)
            mins.append(dist.min(dim=1).values)
        return torch.cat(mins, dim=0)

    @torch.no_grad()
    def predict_batch(self, images: torch.Tensor, output_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        features = self.feature_extractor(images)
        batch_size, _, h, w = features.shape
        patches = self._features_to_patches(features)
        patch_scores = self._min_distances(patches).reshape(batch_size, h, w)
        maps = F.interpolate(patch_scores[:, None, :, :], size=output_size, mode="bilinear", align_corners=False)[:, 0]
        flat = maps.reshape(batch_size, -1)
        k = max(1, int(flat.shape[1] * self.score_topk_ratio))
        image_scores = flat.topk(k, dim=1).values.mean(dim=1)
        return {"image_scores": image_scores.detach().cpu(), "anomaly_maps": maps.detach().cpu()}

    def state_dict(self) -> Dict[str, object]:
        if self.memory_bank is None:
            raise RuntimeError("Cannot save an empty memory bank.")
        return {
            "layers": self.layers,
            "max_memory_patches": self.max_memory_patches,
            "distance_chunk_size": self.distance_chunk_size,
            "score_topk_ratio": self.score_topk_ratio,
            "memory_bank": self.memory_bank.cpu(),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "PatchCoreLite":
        state = torch.load(path, map_location="cpu")
        model = cls(
            layers=state["layers"],
            max_memory_patches=state["max_memory_patches"],
            distance_chunk_size=state["distance_chunk_size"],
            score_topk_ratio=state["score_topk_ratio"],
            device=device,
        )
        model.memory_bank = state["memory_bank"].contiguous()
        return model
