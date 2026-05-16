"""Simple YOLOv8 training stub using ultralytics.

Usage:
    python src/train_yolo.py --data data/visdrone_subset/data.yaml --weights yolov8n.pt --epochs 3
"""
import argparse
from ultralytics import YOLO

from augmentation import get_preset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/visdrone_subset/data.yaml')
    p.add_argument('--weights', default='yolov8n.pt')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--device', default='cpu', help='training device, for example cpu or 0')
    p.add_argument('--augment-preset', default='cpu-lite', choices=['cpu-lite', 'balanced', 'aggressive'])
    p.add_argument('--mosaic', type=float, default=None)
    p.add_argument('--mixup', type=float, default=None)
    p.add_argument('--copy-paste', dest='copy_paste', type=float, default=None)
    p.add_argument('--degrees', type=float, default=None)
    p.add_argument('--translate', type=float, default=None)
    p.add_argument('--scale', type=float, default=None)
    p.add_argument('--shear', type=float, default=None)
    p.add_argument('--perspective', type=float, default=None)
    p.add_argument('--hsv-h', dest='hsv_h', type=float, default=None)
    p.add_argument('--hsv-s', dest='hsv_s', type=float, default=None)
    p.add_argument('--hsv-v', dest='hsv_v', type=float, default=None)
    p.add_argument('--fliplr', type=float, default=None)
    p.add_argument('--flipud', type=float, default=None)
    p.add_argument('--close-mosaic', dest='close_mosaic', type=int, default=None)
    args = p.parse_args()

    print('Starting training with', args)
    model = YOLO(args.weights)
    aug = get_preset(args.augment_preset)
    for key in aug:
        override = getattr(args, key, None)
        if override is not None:
            aug[key] = override

    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
    }
    train_kwargs.update(aug)
    model.train(**train_kwargs)


if __name__ == '__main__':
    main()
