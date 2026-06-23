from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .data import imagenet_transform
from .patchcore_lite import PatchCoreLite
from .utils import ensure_dir, load_config, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video hazelnut defect sorter.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--category", default="hazelnut")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="artifacts/demo/hazelnut_sorter_output.mp4")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--process-every", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-area", type=int, default=500)
    parser.add_argument("--max-area", type=int, default=120000)
    parser.add_argument("--margin", type=float, default=0.20)
    parser.add_argument("--h-min", type=int, default=0)
    parser.add_argument("--h-max", type=int, default=35)
    parser.add_argument("--s-min", type=int, default=35)
    parser.add_argument("--v-min", type=int, default=20)
    parser.add_argument("--debug-dir", default="artifacts/demo/debug_hazelnut")
    parser.add_argument("--save-defect-crops", action="store_true")
    return parser.parse_args()


def detect_hazelnuts_brown_hsv(
    frame_bgr: np.ndarray,
    h_min: int,
    h_max: int,
    s_min: int,
    v_min: int,
    min_area: int,
    max_area: int,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper = np.array([h_max, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    h_img, w_img = frame_bgr.shape[:2]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w < 12 or h < 12:
            continue

        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 2.8:
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        boxes.append((x1, y1, x2, y2))

    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes, mask


def crop_with_margin(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    margin_ratio: float,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    h_img, w_img = frame_bgr.shape[:2]

    w = x2 - x1
    h = y2 - y1
    margin = int(max(w, h) * margin_ratio)

    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)
    cx2 = min(w_img, x2 + margin)
    cy2 = min(h_img, y2 + margin)

    crop = frame_bgr[cy1:cy2, cx1:cx2]
    return crop


def bgr_crop_to_tensor(crop_bgr: np.ndarray, image_size: int, transform) -> torch.Tensor:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb).convert("RGB")
    return transform(image)


def draw_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str,
    score: float,
    is_defect: bool,
    idx: int,
) -> None:
    x1, y1, x2, y2 = box
    color = (0, 0, 255) if is_defect else (0, 180, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"#{idx} {label} {score:.2f}"
    y_text = max(20, y1 - 8)

    cv2.rectangle(frame, (x1, y_text - 18), (x1 + 145, y_text + 4), color, -1)
    cv2.putText(
        frame,
        text,
        (x1 + 4, y_text),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_seen: int,
    total_defects: int,
    threshold: float,
    fps_est: float,
) -> None:
    reject_rate = 100.0 * total_defects / max(total_seen, 1)

    lines = [
        "DefectVision-TRT | Hazelnut Video Sorter",
        f"Frame: {frame_idx}",
        f"Inspected crops: {total_seen}",
        f"Defect alerts: {total_defects} ({reject_rate:.1f}%)",
        f"Threshold: {threshold:.2f}",
        f"Processing FPS: {fps_est:.1f}",
    ]

    x, y = 18, 28
    box_w = 430
    box_h = 26 * len(lines) + 12

    cv2.rectangle(frame, (x - 8, y - 24), (x + box_w, y - 24 + box_h), (20, 20, 20), -1)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    image_size = int(cfg["data"]["image_size"])
    device = resolve_device(cfg["train"].get("device", "auto"))

    artifact_dir = Path("artifacts") / args.category
    model_path = artifact_dir / "patchcore_lite.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"[DefectVision-TRT] Loading model: {model_path}")
    print(f"[DefectVision-TRT] Device: {device}")

    model = PatchCoreLite.load(model_path, device=device)
    transform = imagenet_transform(image_size)

    input_path = Path(args.input)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)

    debug_dir = Path(args.debug_dir)
    ensure_dir(debug_dir)

    crops_dir = output_path.parent / "defect_crops"
    if args.save_defect_crops:
        ensure_dir(crops_dir)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 1:
        video_fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps,
        (width, height),
    )

    scores_csv = output_path.with_suffix(".scores.csv")
    csv_file = open(scores_csv, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "crop_id", "x1", "y1", "x2", "y2", "score", "decision"])

    frame_idx = 0
    total_seen = 0
    total_defects = 0
    last_results: list[tuple[tuple[int, int, int, int], float, bool]] = []
    start_time = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        if frame_idx % args.process_every == 0:
            boxes, mask = detect_hazelnuts_brown_hsv(
                frame,
                h_min=args.h_min,
                h_max=args.h_max,
                s_min=args.s_min,
                v_min=args.v_min,
                min_area=args.min_area,
                max_area=args.max_area,
            )

            if frame_idx % 100 == 0:
                cv2.imwrite(str(debug_dir / f"frame_{frame_idx:06d}.jpg"), frame)
                cv2.imwrite(str(debug_dir / f"mask_{frame_idx:06d}.png"), mask)

            tensors = []
            valid_boxes = []

            for box in boxes:
                crop = crop_with_margin(frame, box, args.margin)
                if crop.size == 0:
                    continue
                tensor = bgr_crop_to_tensor(crop, image_size, transform)
                tensors.append(tensor)
                valid_boxes.append(box)

            current_results: list[tuple[tuple[int, int, int, int], float, bool]] = []

            if tensors:
                batch = torch.stack(tensors, dim=0)
                pred = model.predict_batch(batch, output_size=(image_size, image_size))
                image_scores = pred["image_scores"].numpy()

                for crop_id, (box, score) in enumerate(zip(valid_boxes, image_scores)):
                    score_f = float(score)
                    is_defect = score_f >= args.threshold
                    decision = "DEFECT" if is_defect else "OK"

                    total_seen += 1
                    if is_defect:
                        total_defects += 1

                        if args.save_defect_crops:
                            crop = crop_with_margin(frame, box, args.margin)
                            crop_path = crops_dir / f"frame_{frame_idx:06d}_crop_{crop_id:03d}_score_{score_f:.3f}.jpg"
                            cv2.imwrite(str(crop_path), crop)

                    x1, y1, x2, y2 = box
                    csv_writer.writerow([frame_idx, crop_id, x1, y1, x2, y2, f"{score_f:.6f}", decision])
                    current_results.append((box, score_f, is_defect))

            last_results = current_results

        for idx, (box, score, is_defect) in enumerate(last_results):
            draw_box(
                frame,
                box,
                label="DEFECT" if is_defect else "OK",
                score=score,
                is_defect=is_defect,
                idx=idx,
            )

        elapsed = time.perf_counter() - start_time
        fps_est = (frame_idx + 1) / max(elapsed, 1e-6)

        draw_hud(
            frame,
            frame_idx=frame_idx,
            total_seen=total_seen,
            total_defects=total_defects,
            threshold=args.threshold,
            fps_est=fps_est,
        )

        writer.write(frame)
        frame_idx += 1

    csv_file.close()
    writer.release()
    cap.release()

    print(f"Saved annotated video: {output_path}")
    print(f"Saved crop scores: {scores_csv}")
    print(f"Saved debug frames/masks: {debug_dir}")
    print(f"Frames processed: {frame_idx}")
    print(f"Inspected crops: {total_seen}")
    print(f"Defect alerts: {total_defects}")


if __name__ == "__main__":
    main()