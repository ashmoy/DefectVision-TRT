# Portfolio checklist

Before sharing this project with recruiters, make sure the repository contains:

- [ ] Clean README with problem statement and architecture diagram.
- [ ] At least 3 trained categories: `bottle`, `hazelnut`, `metal_nut`.
- [ ] `metrics.json` for each category.
- [ ] 10+ visual examples with heatmaps and ground-truth masks.
- [ ] ONNX export file generation documented.
- [ ] Benchmark table: PyTorch vs ONNX Runtime vs TensorRT FP16.
- [ ] Short demo video or GIF in the README.
- [ ] Dockerfile or clear environment instructions.
- [ ] CV bullet written in English.

## Benchmark table template

| Category | Backend | Precision | Batch | Mean latency | p95 latency | FPS | Image AUROC | Pixel AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | PyTorch | FP32 | 8 | TODO | TODO | TODO | TODO | TODO |
| bottle | ONNX Runtime | FP32 | 8 | TODO | TODO | TODO | TODO | TODO |
| bottle | TensorRT | FP16 | 8 | TODO | TODO | TODO | TODO | TODO |

## Demo video structure

1. Show a normal industrial image.
2. Show a defective image.
3. Show anomaly heatmap overlay.
4. Show benchmark table.
5. Show TensorRT command and latency gain.
