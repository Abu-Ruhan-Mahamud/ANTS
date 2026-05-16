"""Create a small VisDrone subset for quick experiments.

Usage:
    python src/prepare_subset.py --root VisDrone_Dataset/VisDrone2019-DET-train --out data/visdrone_subset --train 400 --val 100
"""
from pathlib import Path
import argparse
import shutil
import yaml


def load_visdrone_config(root):
    cfg_path = Path(root).parent / 'visdrone.yaml'
    if cfg_path.exists():
        with cfg_path.open('r', encoding='utf-8') as fh:
            return yaml.safe_load(fh)
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='VisDrone_Dataset/VisDrone2019-DET-train')
    p.add_argument('--out', default='data/visdrone_subset')
    p.add_argument('--train', type=int, default=400)
    p.add_argument('--val', type=int, default=100)
    args = p.parse_args()

    root = Path(args.root)
    images_dir = root / 'images'
    labels_dir = root / 'labels'

    out = Path(args.out)
    out_images_train = out / 'images' / 'train'
    out_images_val = out / 'images' / 'val'
    out_labels_train = out / 'labels' / 'train'
    out_labels_val = out / 'labels' / 'val'

    for pth in [out_images_train, out_images_val, out_labels_train, out_labels_val]:
        pth.mkdir(parents=True, exist_ok=True)

    imgs = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png')])
    if len(imgs) == 0:
        print('No images found in', images_dir)
        return

    n_train = min(args.train, len(imgs))
    n_val = min(args.val, max(0, len(imgs) - n_train))

    train_imgs = imgs[:n_train]
    val_imgs = imgs[n_train:n_train + n_val]

    def copy_set(img_list, out_img_dir, out_lbl_dir):
        for im in img_list:
            shutil.copy(im, out_img_dir / im.name)
            lbl = labels_dir / (im.with_suffix('.txt').name)
            if lbl.exists():
                shutil.copy(lbl, out_lbl_dir / lbl.name)

    copy_set(train_imgs, out_images_train, out_labels_train)
    copy_set(val_imgs, out_images_val, out_labels_val)

    # create dataset yaml
    viscfg = load_visdrone_config(root)
    data_yaml = {
        'path': str(out),
        'train': 'images/train',
        'val': 'images/val',
        'nc': viscfg.get('nc', 10) if viscfg else 10,
        'names': viscfg.get('names', {}) if viscfg else {}
    }
    yaml_path = out / 'data.yaml'
    with yaml_path.open('w', encoding='utf-8') as fh:
        yaml.safe_dump(data_yaml, fh)

    print('Created subset at', out)
    print('Train images:', len(train_imgs), 'Val images:', len(val_imgs))
    print('Dataset yaml:', yaml_path)


if __name__ == '__main__':
    main()
