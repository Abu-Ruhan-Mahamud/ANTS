"""Run pretrained YOLOv8 inference on VisDrone images, count humans and cars, save visualizations.

Usage:
    python src/infer_and_count.py --src VisDrone_Dataset/VisDrone2019-DET-train/images --out outputs/sample_results --max 100
"""
from pathlib import Path
import argparse
from ultralytics import YOLO
import cv2
from tqdm import tqdm


COCO_PERSON = 0
COCO_CAR = 2


def parse_class_list(arg_value):
    if not arg_value:
        return None
    return [x.strip().lower() for x in arg_value.split(',') if x.strip()]


def resolve_target_classes(names, human_tokens=None, car_tokens=None):
    id_to_name = {int(k): str(v).lower() for k, v in names.items()}
    name_to_id = {v: k for k, v in id_to_name.items()}

    if human_tokens is None:
        human_tokens = ['person', 'pedestrian', 'people']
    if car_tokens is None:
        car_tokens = ['car']

    human_ids = [name_to_id[t] for t in human_tokens if t in name_to_id]
    car_ids = [name_to_id[t] for t in car_tokens if t in name_to_id]

    # Fallback for COCO defaults
    if not human_ids and COCO_PERSON in id_to_name:
        human_ids = [COCO_PERSON]
    if not car_ids and COCO_CAR in id_to_name:
        car_ids = [COCO_CAR]

    return set(human_ids), set(car_ids)


def draw_boxes(img, boxes, scores, classes, names):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        label = f"{names[int(c)]}:{s:.2f}"
        color = (0, 255, 0) if int(c) == COCO_PERSON else (255, 0, 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(img, label, (int(x1), max(15, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def draw_counts(img, human_count, car_count):
    text = f"Humans: {human_count} | Cars: {car_count}"
    cv2.rectangle(img, (8, 8), (300, 36), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='VisDrone_Dataset/VisDrone2019-DET-train/images')
    p.add_argument('--out', default='outputs/sample_results')
    p.add_argument('--max', type=int, default=50)
    p.add_argument('--weights', default='yolov8n.pt', help='path to model weights (.pt)')
    p.add_argument('--device', default='cpu', help='inference device, for example cpu or 0')
    p.add_argument('--human-classes', default=None, help='comma separated class names to count as humans')
    p.add_argument('--car-classes', default=None, help='comma separated class names to count as cars')
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    img_paths = sorted([p for p in src.iterdir() if p.suffix.lower() in ('.jpg', '.png')])[: args.max]
    if len(img_paths) == 0:
        print('No images found in', src)
        return

    print(f"Loading YOLO model from {args.weights}")
    model = YOLO(args.weights)

    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else {0: 'person', 2: 'car'}
    human_ids, car_ids = resolve_target_classes(
        names,
        parse_class_list(args.human_classes),
        parse_class_list(args.car_classes),
    )
    print(f"Using class ids for humans: {sorted(human_ids)} | cars: {sorted(car_ids)}")

    results_summary = []
    for img_p in tqdm(img_paths, desc='Running inference'):
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        res = model.predict(source=str(img_p), imgsz=640, conf=0.25, iou=0.45, device=args.device, verbose=False)
        # res is a list-like; take first
        r = res[0]
        boxes = []
        scores = []
        classes = []
        human_count = 0
        car_count = 0
        if r.boxes is not None and len(r.boxes) > 0:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy()
                conf = float(b.conf[0].cpu().numpy())
                cls = int(b.cls[0].cpu().numpy())
                if cls in human_ids or cls in car_ids:
                    boxes.append(xyxy)
                    scores.append(conf)
                    classes.append(cls)
                    if cls in human_ids:
                        human_count += 1
                    elif cls in car_ids:
                        car_count += 1

        out_img = draw_boxes(img.copy(), boxes, scores, classes, names)
        out_img = draw_counts(out_img, human_count, car_count)
        out_path = out / img_p.name
        cv2.imwrite(str(out_path), out_img)
        results_summary.append({'image': img_p.name, 'humans': human_count, 'cars': car_count})

    # Save CSV summary
    import csv
    csvp = out / 'counts_summary.csv'
    with csvp.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['image', 'humans', 'cars'])
        writer.writeheader()
        for r in results_summary:
            writer.writerow(r)

    print('Done. Outputs saved to', out)


if __name__ == '__main__':
    main()
