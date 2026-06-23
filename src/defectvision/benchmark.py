from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch

from .feature_extractor import MultiLayerFeatureExtractor
from .utils import ensure_dir, load_config, resolve_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark feature extractor inference.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def summarize(latencies_ms: List[float], batch_size: int) -> Dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "latency_mean_ms": float(arr.mean()),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "fps": float(batch_size * 1000.0 / arr.mean()),
    }


def bench_torch(model, x, warmup: int, iters: int, device: torch.device) -> Dict[str, float]:
    model.eval()
    latencies = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    return summarize(latencies, x.shape[0])


def bench_onnx(onnx_path: Path, x_np: np.ndarray, warmup: int, iters: int) -> Dict[str, float]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    latencies = []
    for _ in range(warmup):
        _ = sess.run(None, {"input": x_np})
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = sess.run(None, {"input": x_np})
        latencies.append((time.perf_counter() - t0) * 1000)
    result = summarize(latencies, x_np.shape[0])
    result["providers"] = providers
    return result


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.category:
        cfg["data"]["category"] = args.category

    category = cfg["data"]["category"]
    image_size = cfg["data"]["image_size"]
    device = resolve_device(cfg["train"].get("device", "auto"))
    out_dir = ensure_dir(Path("artifacts") / category)
    onnx_path = out_dir / "feature_extractor.onnx"

    x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
    model = MultiLayerFeatureExtractor(cfg["model"]["layers"]).to(device).eval()

    results: Dict[str, object] = {
        "category": category,
        "image_size": image_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch": bench_torch(model, x, args.warmup, args.iters, device),
    }

    if onnx_path.exists():
        results["onnxruntime"] = bench_onnx(onnx_path, x.detach().cpu().numpy(), args.warmup, args.iters)
    else:
        results["onnxruntime"] = "Skipped because feature_extractor.onnx does not exist. Run export_onnx first."

    if device.type == "cuda":
        results["gpu_name"] = torch.cuda.get_device_name(0)
        results["max_memory_allocated_mb"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)

    save_json(results, out_dir / "benchmark.json")
    print(results)
    print(f"Saved benchmark: {out_dir / 'benchmark.json'}")
    print("TensorRT next step: run trtexec with the command shown in README.md")


if __name__ == "__main__":
    main()
