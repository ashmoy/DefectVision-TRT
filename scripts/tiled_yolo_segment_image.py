import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def nms(boxes, scores, iou_threshold):
    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        idx = order[0]
        keep.append(idx)

        remaining = []
        for j in order[1:]:
            if iou(boxes[idx], boxes[j]) < iou_threshold:
                remaining.append(j)

        order = np.array(remaining)

    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default="runs/tiled_test/output.jpg")
    parser.add_argument("--tile", type=int, default=320)
    parser.add_argument("--overlap", type=float, default=0.35)
    parser.add_argument("--conf", type=float, default=0.03)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-det", type=int, default=1000)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    image = cv2.imread(args.image)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    h, w = image.shape[:2]

    stride = int(args.tile * (1.0 - args.overlap))
    stride = max(32, stride)

    all_boxes = []
    all_scores = []

    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            x1 = min(x0 + args.tile, w)
            y1 = min(y0 + args.tile, h)

            tile = image[y0:y1, x0:x1]

            if tile.shape[0] < 40 or tile.shape[1] < 40:
                continue

            results = model.predict(
                source=tile,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                device=0,
                verbose=False,
                retina_masks=True,
            )[0]

            if results.boxes is None:
                continue

            boxes = results.boxes.xyxy.detach().cpu().numpy()
            scores = results.boxes.conf.detach().cpu().numpy()

            for box, score in zip(boxes, scores):
                bx1, by1, bx2, by2 = box

                gx1 = int(bx1 + x0)
                gy1 = int(by1 + y0)
                gx2 = int(bx2 + x0)
                gy2 = int(by2 + y0)

                bw = gx2 - gx1
                bh = gy2 - gy1

                if bw < 8 or bh < 8:
                    continue

                if bw > 120 or bh > 120:
                    continue

                all_boxes.append([gx1, gy1, gx2, gy2])
                all_scores.append(float(score))

    if not all_boxes:
        cv2.imwrite(str(out_path), image)
        print("No detections.")
        return

    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)

    keep = nms(all_boxes, all_scores, iou_threshold=0.35)

    vis = image.copy()

    for idx in keep:
        x1, y1, x2, y2 = all_boxes[idx]
        score = all_scores[idx]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            vis,
            f"{score:.2f}",
            (x1, max(15, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), vis)
    print(f"Saved: {out_path}")
    print(f"Raw detections: {len(all_boxes)}")
    print(f"After NMS: {len(keep)}")


if __name__ == "__main__":
    main()