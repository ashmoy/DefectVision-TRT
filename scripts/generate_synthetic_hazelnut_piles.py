import argparse
import json
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def v2_image_paths(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def v2_extract_cutouts(src_root, split):
    """Keep independent cutout pools so train never sees validation/test nuts."""
    cutouts = []
    image_dir = src_root / split / "images"
    label_dir = src_root / split / "labels"

    for image_path in v2_image_paths(image_dir):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read source image: {image_path}")

        height, width = image.shape[:2]
        polygons = read_yolo_seg(label_dir / f"{image_path.stem}.txt", width, height)
        for polygon in polygons:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 255)
            x, y, box_width, box_height = cv2.boundingRect(polygon)
            if box_width < 12 or box_height < 12 or cv2.countNonZero(mask) < 100:
                continue

            pad = 3
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2 = min(width, x + box_width + pad)
            y2 = min(height, y + box_height + pad)
            cutouts.append(
                (image[y1:y2, x1:x2].copy(), mask[y1:y2, x1:x2].copy())
            )

    if not cutouts:
        raise RuntimeError(f"No usable cutouts for split '{split}' in {src_root}")
    print(f"{split}: {len(cutouts)} source cutouts")
    return cutouts


def v2_largest_component(mask):
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask), 0.0

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    return largest, float(areas.max() / max(1, areas.sum()))


def v2_rotate_bound(image, mask, angle):
    height, width = image.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    bound_width = max(1, int(height * sine + width * cosine))
    bound_height = max(1, int(height * cosine + width * sine))
    matrix[0, 2] += bound_width / 2.0 - center[0]
    matrix[1, 2] += bound_height / 2.0 - center[1]
    image = cv2.warpAffine(
        image,
        matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask = cv2.warpAffine(
        mask,
        matrix,
        (bound_width, bound_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image, mask


def v2_transform_cutout(cutout, target_size, rng, np_rng):
    image, mask = cutout
    height, width = image.shape[:2]
    scale = target_size / max(1, width, height)
    new_width = max(8, round(width * scale))
    new_height = max(8, round(height * scale))
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0] + rng.uniform(-7.0, 4.0), 180.0)
    hsv[:, :, 1] *= rng.uniform(0.42, 1.05)
    hsv[:, :, 2] *= rng.uniform(0.55, 1.02)
    image = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    image = np.clip(
        image.astype(np.float32) * rng.uniform(0.90, 1.10)
        + rng.uniform(-9.0, 7.0)
        + np_rng.normal(0, rng.uniform(0.5, 3.0), image.shape),
        0,
        255,
    ).astype(np.uint8)
    if rng.random() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), rng.uniform(0.25, 0.8))

    image, mask = v2_rotate_bound(image, mask, rng.uniform(-180.0, 180.0))
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    image[mask == 0] = 0
    return image, mask


def v2_brown_ratio(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    brown = (
        (hue >= 3)
        & (hue <= 30)
        & (saturation > 35)
        & (value > 35)
        & (value < 230)
    )
    return float(brown.mean())


def v2_crop_to_aspect(image, width, height, rng):
    image_height, image_width = image.shape[:2]
    target_ratio = width / height
    if image_width / image_height > target_ratio:
        crop_width = max(1, round(image_height * target_ratio))
        x1 = rng.randint(0, max(0, image_width - crop_width))
        image = image[:, x1 : x1 + crop_width]
    else:
        crop_height = max(1, round(image_width / target_ratio))
        y1 = rng.randint(0, max(0, image_height - crop_height))
        image = image[y1 : y1 + crop_height, :]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def v2_partition_backgrounds(background_root, seed):
    pools = {"train": [], "valid": [], "test": []}
    if background_root is None:
        return pools

    split_layout = False
    for split in pools:
        paths = v2_image_paths(background_root / split / "images")
        if not paths:
            paths = v2_image_paths(background_root / split)
        if paths:
            pools[split] = paths
            split_layout = True
    if split_layout:
        return pools

    paths = sorted(
        path
        for path in background_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(seed).shuffle(paths)
    train_end = int(len(paths) * 0.8)
    valid_end = train_end + int(len(paths) * 0.1)
    pools["train"] = paths[:train_end]
    pools["valid"] = paths[train_end:valid_end]
    pools["test"] = paths[valid_end:]
    return pools


def v2_real_background(paths, width, height, max_brown_ratio, rng):
    if not paths:
        return None
    for _ in range(min(24, max(1, len(paths) * 2))):
        image = cv2.imread(str(rng.choice(paths)))
        if image is None:
            continue
        image = v2_crop_to_aspect(image, width, height, rng)
        if v2_brown_ratio(image) <= max_brown_ratio:
            return image
    return None


def v2_industrial_background(width, height, rng, np_rng):
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = xx / max(1, width - 1)
    y_norm = yy / max(1, height - 1)
    base = np.empty((height, width, 3), dtype=np.float32)
    base[:, :, 0] = rng.uniform(42, 68) + 9 * x_norm - 4 * y_norm
    base[:, :, 1] = rng.uniform(105, 145) + 18 * x_norm - 8 * y_norm
    base[:, :, 2] = rng.uniform(38, 67) + 7 * x_norm - 5 * y_norm
    noise = np_rng.normal(0, rng.uniform(2.0, 5.0), (height, width, 1))
    canvas = np.clip(base + noise, 0, 255).astype(np.uint8)

    channel = np.array(
        [[round(width * 0.07), 0], [round(width * 0.34), 0],
         [round(width * 0.42), height - 1], [round(width * 0.01), height - 1]],
        np.int32,
    )
    overlay = canvas.copy()
    cv2.fillPoly(
        overlay,
        [channel],
        (rng.randint(38, 61), rng.randint(88, 122), rng.randint(34, 58)),
    )
    canvas = cv2.addWeighted(canvas, 0.35, overlay, 0.65, 0)

    rail = np.array(
        [[round(width * 0.34), 0], [round(width * 0.44), 0],
         [round(width * 0.55), height - 1], [round(width * 0.42), height - 1]],
        np.int32,
    )
    cv2.fillPoly(canvas, [rail], (65, rng.randint(135, 177), 61))
    cv2.polylines(canvas, [rail], True, (42, 92, 43), max(1, width // 480))

    panel = np.array(
        [[round(width * 0.50), round(height * 0.05)], [round(width * 0.93), round(height * 0.02)],
         [width - 1, round(height * 0.55)], [round(width * 0.57), round(height * 0.72)]],
        np.int32,
    )
    cv2.fillPoly(canvas, [panel], (45, rng.randint(82, 120), 43))
    cv2.polylines(canvas, [panel], True, (28, 63, 30), max(1, width // 360))

    hole_center = (
        round(width * rng.uniform(0.69, 0.85)),
        round(height * rng.uniform(0.23, 0.47)),
    )
    hole_radius = round(min(width, height) * rng.uniform(0.06, 0.11))
    cv2.circle(canvas, hole_center, hole_radius, (14, 25, 14), -1)
    cv2.circle(canvas, hole_center, hole_radius, (54, 91, 56), max(2, width // 240))

    for _ in range(rng.randint(3, 8)):
        center = (
            rng.randint(round(width * 0.43), width - 8),
            rng.randint(8, height - 8),
        )
        radius = rng.randint(max(2, width // 420), max(3, width // 220))
        value = rng.randint(70, 150)
        cv2.circle(canvas, center, radius, (value, value, value), -1)
        cv2.circle(canvas, center, radius, (35, 50, 35), 1)

    if rng.random() < 0.8:
        white = np.array(
            [[0, round(height * rng.uniform(0.76, 0.90))],
             [round(width * rng.uniform(0.15, 0.29)), round(height * rng.uniform(0.70, 0.86))],
             [round(width * rng.uniform(0.32, 0.45)), height - 1], [0, height - 1]],
            np.int32,
        )
        shade = rng.randint(205, 245)
        cv2.fillPoly(canvas, [white], (shade, shade, shade))

    for _ in range(rng.randint(5, 15)):
        x1, y1 = rng.randint(0, width - 1), rng.randint(0, height - 1)
        length = rng.randint(max(4, width // 80), max(6, width // 18))
        color = (rng.randint(45, 105), rng.randint(95, 175), rng.randint(42, 95))
        cv2.line(
            canvas,
            (x1, y1),
            (min(width - 1, x1 + length), min(height - 1, y1 + rng.randint(-4, 4))),
            color,
            1,
        )
    return cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.15, 0.55))


def v2_channel_bounds(y_center, width, height):
    t = np.clip(y_center / max(1.0, height - 1.0), 0.0, 1.0)
    left = width * (0.07 * (1.0 - t) + 0.01 * t)
    right = width * (0.34 * (1.0 - t) + 0.42 * t)
    return left, right


def v2_paste(canvas, image, mask, x, y, rng):
    height, width = image.shape[:2]
    roi = canvas[y : y + height, x : x + width]
    if rng.random() < 0.85:
        sigma = max(1.0, min(width, height) * 0.06)
        shadow = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        shadow = np.roll(shadow, (rng.randint(1, 4), rng.randint(1, 4)), (0, 1))
        strength = (shadow.astype(np.float32) / 255.0 * rng.uniform(0.10, 0.28))[..., None]
        roi[:] = np.clip(roi.astype(np.float32) * (1.0 - strength), 0, 255).astype(np.uint8)
    alpha = cv2.GaussianBlur(mask, (0, 0), sigmaX=0.55).astype(np.float32)
    alpha = (alpha / 255.0)[..., None]
    roi[:] = np.clip(
        roi.astype(np.float32) * (1.0 - alpha) + image.astype(np.float32) * alpha,
        0,
        255,
    ).astype(np.uint8)


def v2_mask_to_yolo(mask, x_offset, y_offset, width, height, min_mask_area):
    mask, _ = v2_largest_component(mask)
    if cv2.countNonZero(mask) < min_mask_area:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    polygon = cv2.approxPolyDP(contour, 0.004 * cv2.arcLength(contour, True), True)
    if len(polygon) < 3:
        return None
    coords = []
    for x, y in polygon[:, 0, :]:
        coords.extend(
            [
                np.clip((x + x_offset) / width, 0.0, 1.0),
                np.clip((y + y_offset) / height, 0.0, 1.0),
            ]
        )
    return "0 " + " ".join(f"{value:.6f}" for value in coords)


def v2_generate_one(cutouts, backgrounds, args, rng, np_rng):
    canvas = v2_real_background(
        backgrounds,
        args.width,
        args.height,
        args.max_background_brown_ratio,
        rng,
    )
    real_background_used = canvas is not None
    if canvas is None:
        canvas = v2_industrial_background(args.width, args.height, rng, np_rng)

    if rng.random() < args.negative_fraction:
        return canvas, [], {"requested": 0, "placed": 0, "negative": True, "real_background": real_background_used}

    requested = rng.randint(args.min_nuts, args.max_nuts)
    instances = []
    for _ in range(requested):
        for _ in range(args.max_placement_attempts):
            image, mask = v2_transform_cutout(
                rng.choice(cutouts),
                rng.uniform(args.min_object_px, args.max_object_px),
                rng,
                np_rng,
            )
            object_height, object_width = image.shape[:2]
            if object_width >= args.width or object_height >= args.height:
                continue

            y = rng.randint(0, max(0, args.height - object_height))
            left, right = v2_channel_bounds(y + object_height / 2, args.width, args.height)
            x_min = max(0, round(left - object_width * 0.30))
            x_max = min(args.width - object_width, round(right - object_width * 0.70))
            if x_max < x_min:
                continue
            x = rng.randint(x_min, x_max)

            original_area = cv2.countNonZero(mask)
            if original_area < args.min_mask_area:
                continue

            updated_masks = []
            valid = True
            for previous in instances:
                previous_x, previous_y, previous_x2, previous_y2 = previous["bbox"]
                intersection_x1 = max(x, previous_x)
                intersection_y1 = max(y, previous_y)
                intersection_x2 = min(x + object_width, previous_x2)
                intersection_y2 = min(y + object_height, previous_y2)
                if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
                    updated_masks.append(None)
                    continue

                candidate_slice = mask[
                    intersection_y1 - y : intersection_y2 - y,
                    intersection_x1 - x : intersection_x2 - x,
                ]
                if not np.any(candidate_slice):
                    updated_masks.append(None)
                    continue

                updated = previous["mask"].copy()
                previous_slice = updated[
                    intersection_y1 - previous_y : intersection_y2 - previous_y,
                    intersection_x1 - previous_x : intersection_x2 - previous_x,
                ]
                previous_slice[candidate_slice > 0] = 0
                largest, component_ratio = v2_largest_component(updated)
                visible_ratio = cv2.countNonZero(largest) / max(1, previous["original_area"])
                if visible_ratio < args.min_visible_ratio or component_ratio < args.min_component_ratio:
                    valid = False
                    break
                updated_masks.append(largest)
            if not valid:
                continue

            for previous, updated in zip(instances, updated_masks):
                if updated is not None:
                    previous["mask"] = updated
            v2_paste(canvas, image, mask, x, y, rng)
            instances.append(
                {
                    "mask": mask.copy(),
                    "bbox": (x, y, x + object_width, y + object_height),
                    "original_area": original_area,
                }
            )
            break

    labels = []
    for instance in instances:
        x1, y1, _, _ = instance["bbox"]
        line = v2_mask_to_yolo(
            instance["mask"],
            x1,
            y1,
            args.width,
            args.height,
            args.min_mask_area,
        )
        if line is not None:
            labels.append(line)
    return canvas, labels, {
        "requested": requested,
        "placed": len(labels),
        "negative": False,
        "real_background": real_background_used,
    }


def v2_validate_dataset(root, expected_counts):
    summary = {}
    errors = []
    for split, expected in expected_counts.items():
        images = {p.stem: p for p in v2_image_paths(root / split / "images")}
        labels = {p.stem: p for p in (root / split / "labels").glob("*.txt")}
        if len(images) != expected or len(labels) != expected:
            errors.append(
                f"{split}: expected {expected} pairs, got {len(images)} images/{len(labels)} labels"
            )
        if set(images) != set(labels):
            errors.append(f"{split}: image/label stems do not match")

        objects = 0
        empty_images = 0
        for label_path in labels.values():
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_images += 1
                continue
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 7 or (len(parts) - 1) % 2:
                    errors.append(f"Malformed row in {label_path}")
                    continue
                coords = np.asarray(list(map(float, parts[1:])))
                if np.any(coords < 0) or np.any(coords > 1):
                    errors.append(f"Out-of-range coordinates in {label_path}")
                objects += 1
        summary[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_images": empty_images,
            "objects": objects,
        }
    if errors:
        raise RuntimeError("Dataset integrity check failed:\n- " + "\n- ".join(errors[:30]))
    return summary


def v2_parse_args():
    parser = argparse.ArgumentParser(
        description="Generate leak-free synthetic hazelnut instance-segmentation data."
    )
    parser.add_argument("--src", default="data/hazelnut_seg_yolo")
    parser.add_argument("--out", default="data/hazelnut_synth_piles_v2_yolo")
    parser.add_argument("--num", type=int, default=1200)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--size", type=int, default=None, help="Legacy square override")
    parser.add_argument("--min-nuts", type=int, default=55)
    parser.add_argument("--max-nuts", type=int, default=105)
    parser.add_argument("--min-object-px", type=int, default=30)
    parser.add_argument("--max-object-px", type=int, default=56)
    parser.add_argument("--negative-fraction", type=float, default=0.10)
    parser.add_argument("--min-visible-ratio", type=float, default=0.35)
    parser.add_argument("--min-component-ratio", type=float, default=0.90)
    parser.add_argument("--min-mask-area", type=int, default=140)
    parser.add_argument("--max-placement-attempts", type=int, default=30)
    parser.add_argument("--background-dir", type=Path, default=None)
    parser.add_argument("--max-background-brown-ratio", type=float, default=0.04)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = v2_parse_args()
    if args.size is not None:
        args.width = args.height = args.size
    if args.num < 3:
        raise ValueError("--num must be at least 3")
    if not (0 <= args.negative_fraction < 1):
        raise ValueError("--negative-fraction must be in [0, 1)")
    if args.min_nuts < 1 or args.max_nuts < args.min_nuts:
        raise ValueError("Invalid nut-count range")
    if args.min_object_px < 8 or args.max_object_px < args.min_object_px:
        raise ValueError("Invalid object-size range")
    if args.train_ratio <= 0 or args.valid_ratio <= 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("Split ratios must leave a non-empty test split")

    src_root, out_root = Path(args.src), Path(args.out)
    if out_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Output already exists: {out_root}. Choose another --out or pass --overwrite."
            )
        shutil.rmtree(out_root)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)
    cutouts = {
        split: v2_extract_cutouts(src_root, split)
        for split in ("train", "valid", "test")
    }
    background_pools = v2_partition_backgrounds(args.background_dir, args.seed)

    train_count = int(args.num * args.train_ratio)
    valid_count = int(args.num * args.valid_ratio)
    split_counts = {
        "train": train_count,
        "valid": valid_count,
        "test": args.num - train_count - valid_count,
    }
    generation = {}
    global_index = 0
    for split, count in split_counts.items():
        image_dir = out_root / split / "images"
        label_dir = out_root / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        stats = {"requested_objects": 0, "written_objects": 0, "negative_images": 0}

        for _ in range(count):
            image, labels, item_stats = v2_generate_one(
                cutouts[split], background_pools[split], args, rng, np_rng
            )
            stem = f"synth_{global_index:06d}"
            image_path = image_dir / f"{stem}.jpg"
            if not cv2.imwrite(
                str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]
            ):
                raise RuntimeError(f"OpenCV failed to write {image_path}")
            (label_dir / f"{stem}.txt").write_text(
                "\n".join(labels), encoding="utf-8"
            )
            stats["requested_objects"] += item_stats["requested"]
            stats["written_objects"] += item_stats["placed"]
            stats["negative_images"] += int(item_stats["negative"])
            global_index += 1

        generation[split] = stats
        print(
            f"{split}: {count} images, {stats['written_objects']} objects, "
            f"{stats['negative_images']} negatives"
        )

    yaml = f'''path: "{out_root.resolve().as_posix()}"
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["hazelnut"]
'''
    (out_root / "data.yaml").write_text(yaml, encoding="utf-8")
    integrity = v2_validate_dataset(out_root, split_counts)
    manifest = {
        "generator": "generate_synthetic_hazelnut_piles.py v2",
        "seed": args.seed,
        "source": str(src_root.resolve()),
        "background_dir": str(args.background_dir.resolve()) if args.background_dir else None,
        "image_size": [args.width, args.height],
        "split_counts": split_counts,
        "object_size_px": [args.min_object_px, args.max_object_px],
        "nut_count": [args.min_nuts, args.max_nuts],
        "negative_fraction": args.negative_fraction,
        "generation": generation,
        "integrity": integrity,
    }
    (out_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"saved dataset: {out_root}")
    print(json.dumps(integrity, indent=2))


if __name__ == "__main__":
    main()
