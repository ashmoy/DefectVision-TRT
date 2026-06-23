import argparse
from pathlib import Path
import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=int, default=80)
    parser.add_argument("--max-frames", type=int, default=1)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_idx = 0
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.step == 0:
            out_path = out_dir / f"frame_{saved:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1

            if saved >= args.max_frames:
                break

        frame_idx += 1

    cap.release()
    print(f"Saved {saved} frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
