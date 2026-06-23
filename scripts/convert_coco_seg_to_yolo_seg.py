import argparse
import json
import shutil
from pathlib import Path


def find_annotation_file(split_dir: Path) -> Path:
    candidates = list(split_dir.glob("*.json"))
    if not candidates:
        raise FileNotFoundError(f"No COCO JSON annotation file found in {split_dir}")
    for c in candidates:
        if "annotation" in c.name.lower() or "coco" in c.name.lower():
            return c
    return candidates[0]


def polygon_area(poly):
    pts = list(zip(poly[0::2], poly[1::2]))
    if len(pts) < 3:
        return 0.0
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def convert_split(src_root: Path, out_root: Path, split: str):
    split_dir = src_root / split
    if not split_dir.exists():
        print(f"[skip] {split} not found")
        return

    ann_path = find_annotation_file(split_dir)
    print(f"[{split}] using annotations: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    out_images = out_root / split / "images"
    out_labels = out_root / split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_image = {img_id: [] for img_id in images}

    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id in anns_by_image:
            anns_by_image[img_id].append(ann)

    copied = 0
    labels_written = 0
    skipped_rle = 0

    for img_id, img in images.items():
        file_name = img["file_name"]
        width = img["width"]
        height = img["height"]

        src_img = split_dir / file_name
        if not src_img.exists():
            src_img = split_dir / "images" / file_name

        if not src_img.exists():
            print(f"[warn] missing image: {file_name}")
            continue

        dst_img = out_images / Path(file_name).name
        shutil.copy2(src_img, dst_img)
        copied += 1

        label_path = out_labels / (Path(file_name).stem + ".txt")
        lines = []

        for ann in anns_by_image.get(img_id, []):
            seg = ann.get("segmentation")

            if not seg:
                continue

            if isinstance(seg, dict):
                skipped_rle += 1
                continue

            if not isinstance(seg, list):
                continue

            polygons = []
            for poly in seg:
                if isinstance(poly, list) and len(poly) >= 6:
                    polygons.append(poly)

            if not polygons:
                continue

            poly = max(polygons, key=polygon_area)

            coords = []
            for x, y in zip(poly[0::2], poly[1::2]):
                xn = max(0.0, min(1.0, x / width))
                yn = max(0.0, min(1.0, y / height))
                coords.extend([xn, yn])

            if len(coords) >= 6:
                line = "0 " + " ".join(f"{v:.6f}" for v in coords)
                lines.append(line)

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        labels_written += len(lines)

    print(f"[{split}] images copied: {copied}")
    print(f"[{split}] segment labels written: {labels_written}")
    if skipped_rle:
        print(f"[{split}] skipped RLE segmentations: {skipped_rle}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default="hazelnut")
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        convert_split(src_root, out_root, split)

    yaml_path = out_root / "data.yaml"
    yaml_content = f"""path: "{out_root.resolve().as_posix()}"
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["{args.name}"]
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"Saved YOLO segmentation YAML: {yaml_path}")


if __name__ == "__main__":
    main()