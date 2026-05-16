"""Inspect VisDrone dataset: counts, sample annotations, and category distribution.

Usage:
    python src/data_inspect.py --root "./VisDrone_Dataset/VisDrone2019-DET-train"

The script prints image/label counts and shows sample label lines and category id counts.
"""
from pathlib import Path
import argparse
import sys

def parse_label_line(line: str):
    # VisDrone annotation format is CSV-like: x,y,w,h,score,category,occlusion,ignore
    parts = [p.strip() for p in line.replace('\t', ',').split(',') if p.strip()]
    if len(parts) < 6:
        parts = line.split()
    return parts

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='VisDrone_Dataset/VisDrone2019-DET-train', help='dataset train root')
    args = p.parse_args()
    root = Path(args.root)
    images_dir = root / 'images'
    labels_dir = root / 'labels'

    if not root.exists():
        print('Dataset root not found:', root)
        sys.exit(1)

    images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    labels = sorted([p for p in labels_dir.iterdir() if p.suffix.lower() in ('.txt',)])

    print('Dataset root:', root)
    print('Images found:', len(images))
    print('Label files found:', len(labels))

    # Show a few sample images and label lines
    sample_n = 5
    print('\nSample label files and first lines:')
    cat_counts = {}
    for i, lf in enumerate(labels[:sample_n]):
        print('-', lf.name)
        with lf.open('r', encoding='utf-8', errors='ignore') as fh:
            lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
            for ln in lines[:5]:
                parts = parse_label_line(ln)
                print('   ', ln)
                # try to parse category id
                try:
                    cid = int(parts[5])
                    cat_counts[cid] = cat_counts.get(cid, 0) + 1
                except Exception:
                    pass

    # If more label files, aggregate category counts across all labels (fast sample)
    print('\nAggregating category ids across label files (sampled)...')
    for lf in labels[:1000]:
        with lf.open('r', encoding='utf-8', errors='ignore') as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                parts = parse_label_line(ln)
                try:
                    cid = int(parts[5])
                    cat_counts[cid] = cat_counts.get(cid, 0) + 1
                except Exception:
                    continue

    print('\nCategory id counts (sampled):')
    for k in sorted(cat_counts.keys()):
        print(f'  id {k}: {cat_counts[k]}')

    print('\nDone. Next: convert annotations to COCO/Yolo format or visualize samples.')

if __name__ == '__main__':
    main()
