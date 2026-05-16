"""Create sample VisDrone visualizations with ground-truth boxes.

Usage:
    python src/visualize_dataset_samples.py --root VisDrone_Dataset --split val --num 9
"""

from pathlib import Path
import argparse
import random

import cv2
import yaml


def load_names(root: Path):
    cfg_path = root / "visdrone.yaml"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg.get("names", {})
    return {int(k): str(v) for k, v in names.items()}


def draw_label(image, text, x1, y1, color):
    y = max(18, y1 - 6)
    cv2.putText(image, text, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def parse_yolo_label(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(parts[0])
    x, y, w, h = map(float, parts[1:])
    return cls, x, y, w, h


def yolo_to_xyxy(x, y, w, h, img_w, img_h):
    x1 = int((x - w / 2.0) * img_w)
    y1 = int((y - h / 2.0) * img_h)
    x2 = int((x + w / 2.0) * img_w)
    y2 = int((y + h / 2.0) * img_h)
    return x1, y1, x2, y2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="VisDrone_Dataset")
    p.add_argument("--split", default="val", choices=["train", "val"])
    p.add_argument("--num", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="outputs/dataset_samples")
    args = p.parse_args()

    random.seed(args.seed)

    root = Path(args.root)
    split_dir = root / f"VisDrone2019-DET-{args.split}"
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(root)

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")])
    if not image_paths:
        print("No images found in", images_dir)
        return

    picked = random.sample(image_paths, k=min(args.num, len(image_paths)))

    for img_p in picked:
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        h, w = img.shape[:2]
        lbl_p = labels_dir / (img_p.stem + ".txt")

        if lbl_p.exists():
            for line in lbl_p.read_text(encoding="utf-8").splitlines():
                parsed = parse_yolo_label(line)
                if parsed is None:
                    continue
                cls, cx, cy, bw, bh = parsed
                x1, y1, x2, y2 = yolo_to_xyxy(cx, cy, bw, bh, w, h)
                color = (0, 255, 0) if cls in (0, 1) else (255, 128, 0)
                cls_name = names.get(cls, str(cls))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                draw_label(img, cls_name, x1, y1, color)

        out_path = out_dir / img_p.name
        cv2.imwrite(str(out_path), img)

    print("Saved", len(picked), "sample visualizations to", out_dir)


if __name__ == "__main__":
    main()
