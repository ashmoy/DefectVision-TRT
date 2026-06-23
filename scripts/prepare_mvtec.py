from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MVTec AD archive.")
    parser.add_argument("--archive", required=True, help="Path to mvtec_anomaly_detection.tar.xz")
    parser.add_argument("--out", default="data/mvtec", help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = Path(args.archive)
    out = Path(args.out)
    if not archive.exists():
        raise FileNotFoundError(archive)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive} -> {out}")
    with tarfile.open(archive, mode="r:xz") as tar:
        tar.extractall(out)

    # Some archives extract into an extra root folder. Flatten if needed.
    children = [p for p in out.iterdir() if p.is_dir()]
    if len(children) == 1 and (children[0] / "bottle").exists():
        root = children[0]
        for child in root.iterdir():
            target = out / child.name
            if target.exists():
                continue
            shutil.move(str(child), str(target))
        try:
            root.rmdir()
        except OSError:
            pass
    print("Done.")


if __name__ == "__main__":
    main()
