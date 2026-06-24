import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def green_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (h >= 35) & (h <= 95) & (s > 35) & (v > 40)
    return float(mask.mean())


def brown_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    mask = (h >= 3) & (h <= 30) & (s > 35) & (v > 35) & (v < 230)
    return float(mask.mean())


def bright_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = (v > 210) & (s < 90)
    return float(mask.mean())


def copy_dataset(src, out):
    src = Path(src)
    out = Path(out)

    if out.exists():
        raise RuntimeError(f"Output already exists: {out}. Choose another --out folder.")

    shutil.copytree(src, out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-yolo", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-neg", type=int, default=500)
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--crops-per-frame", type=int, default=25)
    args = parser.parse_args()

    out_root = copy_dataset(args.src_yolo, args.out)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    train_img_dir = out_root / "train" / "images"
    train_lab_dir = out_root / "train" / "labels"
    valid_img_dir = out_root / "valid" / "images"
    valid_lab_dir = out_root / "valid" / "labels"

    train_img_dir.mkdir(parents=True, exist_ok=True)
    train_lab_dir.mkdir(parents=True, exist_ok=True)
    valid_img_dir.mkdir(parents=True, exist_ok=True)
    valid_lab_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    tried = 0

    while saved < args.num_neg:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.step != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]

        for _ in range(args.crops_per_frame):
            if saved >= args.num_neg:
                break

            crop_size = args.crop
            if w <= crop_size or h <= crop_size:
                continue

            x = random.randint(0, w - crop_size)
            y = random.randint(0, h - crop_size)

            crop = frame[y:y + crop_size, x:x + crop_size]
            tried += 1

            g = green_ratio(crop)
            b = brown_ratio(crop)
            br = bright_ratio(crop)

            # On veut surtout des zones de plastique / fond / reflets,
            # mais pas des crops remplis de noisettes.
            keep = False

            if g > 0.35 and b < 0.25:
                keep = True

            if br > 0.18 and b < 0.25:
                keep = True

            # Zone sombre / fond peu texturé, parfois détectée à tort
            if b < 0.12 and g > 0.20:
                keep = True

            if not keep:
                continue

            split = "valid" if random.random() < 0.15 else "train"

            if split == "train":
                img_dir = train_img_dir
                lab_dir = train_lab_dir
            else:
                img_dir = valid_img_dir
                lab_dir = valid_lab_dir

            name = f"hard_negative_{saved:06d}.jpg"
            label_name = f"hard_negative_{saved:06d}.txt"

            cv2.imwrite(str(img_dir / name), crop)

            # fichier label vide = image sans objet
            (lab_dir / label_name).write_text("", encoding="utf-8")

            saved += 1

        frame_idx += 1

    cap.release()

    print(f"saved hard negatives: {saved}")
    print(f"tried crops: {tried}")
    print(f"output dataset: {out_root}")


if __name__ == "__main__":
    main()