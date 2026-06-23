import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def polygon_to_mask(poly, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly is None or len(poly) < 3:
        return mask
    pts = np.array(poly, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def green_ratio_in_mask(frame, mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]

    if len(pixels) == 0:
        return 1.0

    h = pixels[:, 0]
    s = pixels[:, 1]
    v = pixels[:, 2]

    green = ((h >= 35) & (h <= 95) & (s > 35) & (v > 40))
    return float(green.mean())


def bright_ratio_in_mask(frame, mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]

    if len(pixels) == 0:
        return 1.0

    s = pixels[:, 1]
    v = pixels[:, 2]

    bright = ((v > 210) & (s < 80))
    return float(bright.mean())


def passes_filter(
    frame,
    mask,
    box,
    min_area,
    max_area,
    min_w,
    max_w,
    min_h,
    max_h,
    min_aspect,
    max_aspect,
    min_extent,
    max_green_ratio,
    max_bright_ratio,
    roi,
):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area = int((mask > 0).sum())

    if roi is not None:
        rx1, ry1, rx2, ry2 = roi
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            return False, "roi"

    if area < min_area:
        return False, "small_area"

    if area > max_area:
        return False, "large_area"

    if bw < min_w or bh < min_h:
        return False, "small_box"

    if bw > max_w or bh > max_h:
        return False, "large_box"

    aspect = bw / bh
    if aspect < min_aspect or aspect > max_aspect:
        return False, "aspect"

    extent = area / float(bw * bh)
    if extent < min_extent:
        return False, "low_extent"

    green_ratio = green_ratio_in_mask(frame, mask)
    if green_ratio > max_green_ratio:
        return False, "green"

    bright_ratio = bright_ratio_in_mask(frame, mask)
    if bright_ratio > max_bright_ratio:
        return False, "bright"

    return True, "keep"


def draw_mask_overlay(frame, mask, box, score):
    overlay = frame.copy()
    color = (255, 0, 0)

    overlay[mask > 0] = (
        0.55 * overlay[mask > 0] + 0.45 * np.array(color)
    ).astype(np.uint8)

    x1, y1, x2, y2 = box
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default="runs/filtered/hazelnut_filtered.mp4")

    parser.add_argument("--conf", type=float, default=0.08)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-det", type=int, default=2000)

    parser.add_argument("--min-area", type=int, default=120)
    parser.add_argument("--max-area", type=int, default=2600)
    parser.add_argument("--min-w", type=int, default=8)
    parser.add_argument("--max-w", type=int, default=85)
    parser.add_argument("--min-h", type=int, default=8)
    parser.add_argument("--max-h", type=int, default=85)
    parser.add_argument("--min-aspect", type=float, default=0.45)
    parser.add_argument("--max-aspect", type=float, default=2.20)
    parser.add_argument("--min-extent", type=float, default=0.20)
    parser.add_argument("--max-green-ratio", type=float, default=0.35)
    parser.add_argument("--max-bright-ratio", type=float, default=0.65)

    parser.add_argument("--roi", nargs=4, type=int, default=None)

    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    frame_idx = 0
    total_raw = 0
    total_keep = 0
    reject_stats = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
            device=0,
            retina_masks=True,
            verbose=False,
        )[0]

        vis = frame.copy()

        if result.boxes is not None and result.masks is not None:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            polygons = result.masks.xy

            total_raw += len(boxes)

            kept_this_frame = []

            for box, score, poly in zip(boxes, scores, polygons):
                x1, y1, x2, y2 = box.astype(int)
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w - 1, x2))
                y2 = max(0, min(h - 1, y2))

                mask = polygon_to_mask(poly, h, w)

                keep, reason = passes_filter(
                    frame=frame,
                    mask=mask,
                    box=(x1, y1, x2, y2),
                    min_area=args.min_area,
                    max_area=args.max_area,
                    min_w=args.min_w,
                    max_w=args.max_w,
                    min_h=args.min_h,
                    max_h=args.max_h,
                    min_aspect=args.min_aspect,
                    max_aspect=args.max_aspect,
                    min_extent=args.min_extent,
                    max_green_ratio=args.max_green_ratio,
                    max_bright_ratio=args.max_bright_ratio,
                    roi=args.roi,
                )

                if keep:
                    kept_this_frame.append((mask, (x1, y1, x2, y2), score))
                else:
                    reject_stats[reason] = reject_stats.get(reason, 0) + 1

            total_keep += len(kept_this_frame)

            for mask, box, score in kept_this_frame:
                vis = draw_mask_overlay(vis, mask, box, score)

        writer.write(vis)

        if frame_idx % 50 == 0:
            print(f"frame {frame_idx} | raw={total_raw} | kept={total_keep} | rejects={reject_stats}")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"saved: {out_path}")
    print(f"raw detections: {total_raw}")
    print(f"kept detections: {total_keep}")
    print(f"reject stats: {reject_stats}")


if __name__ == "__main__":
    main()