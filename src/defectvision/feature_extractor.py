from __future__ import annotations

from typing import Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from torchvision.models.feature_extraction import create_feature_extractor


class MultiLayerFeatureExtractor(nn.Module):
    """Extract and concatenate intermediate CNN feature maps.

    PatchCore commonly uses intermediate CNN layers because they preserve local texture
    and part-level information better than a final classification embedding.
    """

    def __init__(self, layers: Iterable[str] = ("layer2", "layer3")) -> None:
        super().__init__()
        weights = Wide_ResNet50_2_Weights.DEFAULT
        backbone = wide_resnet50_2(weights=weights)
        return_nodes = {layer: layer for layer in layers}
        self.extractor = create_feature_extractor(backbone, return_nodes=return_nodes)
        self.layers = list(layers)
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: Dict[str, torch.Tensor] = self.extractor(x)
        maps: List[torch.Tensor] = [features[layer] for layer in self.layers]
        target_size = maps[0].shape[-2:]
        resized = [maps[0]]
        for feature_map in maps[1:]:
            resized.append(F.interpolate(feature_map, size=target_size, mode="bilinear", align_corners=False))
        return torch.cat(resized, dim=1)
