import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_yolo_seg(label_path, w, h):
    polygons = []
    if not label_path.exists():
        return polygons

    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue

        coords = list(map(float, parts[1:]))
        pts = []
        for x, y in zip(coords[0::2], coords[1::2]):
            pts.append([int(x * w), int(y * h)])

        if len(pts) >= 3:
            polygons.append(np.array(pts, dtype=np.int32))

    return polygons


def extract_cutouts(src_root):
    cutouts = []

    for split in ["train", "valid", "test"]:
        image_dir = src_root / split / "images"
        label_dir = src_root / split / "labels"

        if not image_dir.exists():
            continue

        for img_path in image_dir.glob("*.*"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            label_path = label_dir / (img_path.stem + ".txt")
            polygons = read_yolo_seg(label_path, w, h)

            for poly in polygons:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [poly], 255)

                x, y, bw, bh = cv2.boundingRect(poly)
                if bw < 20 or bh < 20:
                    continue

                crop = img[y:y + bh, x:x + bw].copy()
                crop_mask = mask[y:y + bh, x:x + bw].copy()

                if crop.size == 0:
                    continue

                cutouts.append((crop, crop_mask))

    print(f"cutouts extracted: {len(cutouts)}")
    return cutouts


def random_green_background(size):
    h, w = size
    base = np.zeros((h, w, 3), dtype=np.uint8)

    green = random.randint(105, 160)
    base[:, :, 0] = random.randint(35, 80)
    base[:, :, 1] = green
    base[:, :, 2] = random.randint(35, 80)

    noise = np.random.normal(0, 9, (h, w, 3)).astype(np.int16)
    bg = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for _ in range(random.randint(2, 5)):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = random.randint(0, w - 1)
        y2 = random.randint(0, h - 1)
        color = (
            random.randint(40, 90),
            random.randint(130, 210),
            random.randint(40, 90),
        )
        cv2.line(bg, (x1, y1), (x2, y2), color, random.randint(2, 8))

    bg = cv2.GaussianBlur(bg, (3, 3), 0)
    return bg


def transform_cutout(crop, mask):
    scale = random.uniform(0.55, 1.25)
    angle = random.uniform(-180, 180)

    h, w = crop.shape[:2]
    new_w = max(10, int(w * scale))
    new_h = max(10, int(h * scale))

    crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    h, w = crop.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    bound_w = int((h * sin) + (w * cos))
    bound_h = int((h * cos) + (w * sin))

    matrix[0, 2] += bound_w / 2 - center[0]
    matrix[1, 2] += bound_h / 2 - center[1]

    crop = cv2.warpAffine(crop, matrix, (bound_w, bound_h), borderValue=(0, 0, 0))
    mask = cv2.warpAffine(mask, matrix, (bound_w, bound_h), borderValue=0)

    if random.random() < 0.35:
        crop = cv2.GaussianBlur(crop, (3, 3), 0)

    alpha = random.uniform(0.85, 1.15)
    beta = random.randint(-15, 15)
    crop = np.clip(crop.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    return crop, mask


def mask_to_yolo_polygons(mask, img_w, img_h):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80:
            continue

        epsilon = 0.006 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) < 3:
            continue

        coords = []
        for p in approx[:, 0, :]:
            x = max(0.0, min(1.0, p[0] / img_w))
            y = max(0.0, min(1.0, p[1] / img_h))
            coords.extend([x, y])

        if len(coords) >= 6:
            lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))

    return lines


def generate_one(cutouts, out_img_size, min_nuts, max_nuts):
    h, w = out_img_size
    canvas = random_green_background((h, w))

    visible_masks = []
    n = random.randint(min_nuts, max_nuts)

    # zone where nuts mostly appear, like a conveyor channel
    x_min = int(w * random.uniform(0.15, 0.25))
    x_max = int(w * random.uniform(0.65, 0.80))
    y_min = int(h * random.uniform(0.02, 0.08))
    y_max = int(h * random.uniform(0.92, 0.98))

    for _ in range(n):
        crop, mask = random.choice(cutouts)
        crop, mask = transform_cutout(crop, mask)

        ch, cw = crop.shape[:2]
        if ch >= h or cw >= w:
            continue

        x = random.randint(max(0, x_min - cw // 2), min(w - cw, x_max))
        y = random.randint(max(0, y_min - ch // 2), min(h - ch, y_max))

        roi = canvas[y:y + ch, x:x + cw]
        alpha = (mask.astype(np.float32) / 255.0)[..., None]

        roi[:] = (roi.astype(np.float32) * (1 - alpha) + crop.astype(np.float32) * alpha).astype(np.uint8)

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y:y + ch, x:x + cw] = mask

        # occlusion: previous masks lose the pixels covered by the new nut
        for i in range(len(visible_masks)):
            visible_masks[i][full_mask > 0] = 0

        visible_masks.append(full_mask)

    label_lines = []
    for m in visible_masks:
        label_lines.extend(mask_to_yolo_polygons(m, w, h))

    return canvas, label_lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/hazelnut_seg_yolo")
    parser.add_argument("--out", default="data/hazelnut_synth_piles_yolo")
    parser.add_argument("--num", type=int, default=600)
    parser.add_argument("--size", type=int, default=960)
    parser.add_argument("--min-nuts", type=int, default=45)
    parser.add_argument("--max-nuts", type=int, default=100)
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    if out_root.exists():
        shutil.rmtree(out_root)

    cutouts = extract_cutouts(src_root)
    if not cutouts:
        raise RuntimeError("No cutouts found. Check your YOLO segmentation dataset.")

    splits = {
        "train": int(args.num * 0.8),
        "valid": int(args.num * 0.1),
        "test": args.num - int(args.num * 0.8) - int(args.num * 0.1),
    }

    idx_global = 0

    for split, count in splits.items():
        img_dir = out_root / split / "images"
        lab_dir = out_root / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            img, labels = generate_one(
                cutouts,
                out_img_size=(args.size, args.size),
                min_nuts=args.min_nuts,
                max_nuts=args.max_nuts,
            )

            img_name = f"synth_{idx_global:06d}.jpg"
            lab_name = f"synth_{idx_global:06d}.txt"

            cv2.imwrite(str(img_dir / img_name), img)
            (lab_dir / lab_name).write_text("\n".join(labels), encoding="utf-8")

            idx_global += 1

        print(f"{split}: {count} images")

    yaml = f"""path: "{out_root.resolve().as_posix()}"
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["hazelnut"]
"""
    (out_root / "data.yaml").write_text(yaml, encoding="utf-8")
    print(f"saved dataset: {out_root}")


if __name__ == "__main__":
    main()