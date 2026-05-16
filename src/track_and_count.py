"""Track humans and cars with YOLO + tracker (ByteTrack/BoT-SORT).

Usage examples:
    python src/track_and_count.py --src VisDrone_Dataset/VisDrone2019-DET-val/images --weights runs/detect/train-4/weights/best.pt --tracker bytetrack --max 200
    python src/track_and_count.py --src demo.mp4 --weights runs/detect/train-4/weights/best.pt --tracker botsort
"""

from pathlib import Path
import argparse
import csv

import cv2
from ultralytics import YOLO


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
    return set(human_ids), set(car_ids)


def draw_counts(img, frame_humans, frame_cars, unique_humans):
    text = f"Frame Humans: {frame_humans} | Cars: {frame_cars} | Unique Humans: {unique_humans}"
    cv2.rectangle(img, (8, 8), (620, 40), (0, 0, 0), -1)
    cv2.putText(img, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return img


def image_sequence(source: Path):
    return sorted([p for p in source.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])


def run_on_images(args, model, human_ids, car_ids):
    src = Path(args.src)
    frames = image_sequence(src)
    if args.max > 0:
        frames = frames[:args.max]

    if not frames:
        print('No images found in', src)
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if args.save_video:
        first = cv2.imread(str(frames[0]))
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(out_dir / 'tracked.mp4'), fourcc, args.fps, (w, h))

    unique_human_tracks = set()
    csv_rows = []

    tracker_cfg = 'bytetrack.yaml' if args.tracker == 'bytetrack' else 'botsort.yaml'

    for i, img_path in enumerate(frames, start=1):
        result = model.track(
            source=str(img_path),
            persist=True,
            tracker=tracker_cfg,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        frame = result.plot()

        frame_humans = 0
        frame_cars = 0
        human_track_ids = []

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            ids = None
            if result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)

            for idx, cls_id in enumerate(classes):
                if cls_id in human_ids:
                    frame_humans += 1
                    if ids is not None:
                        human_track_ids.append(int(ids[idx]))
                elif cls_id in car_ids:
                    frame_cars += 1

        for tid in human_track_ids:
            unique_human_tracks.add(tid)

        frame = draw_counts(frame, frame_humans, frame_cars, len(unique_human_tracks))
        cv2.imwrite(str(out_dir / img_path.name), frame)
        if writer is not None:
            writer.write(frame)

        csv_rows.append(
            {
                'frame_index': i,
                'image': img_path.name,
                'frame_humans': frame_humans,
                'frame_cars': frame_cars,
                'unique_human_tracks_so_far': len(unique_human_tracks),
            }
        )

    if writer is not None:
        writer.release()

    with (out_dir / 'tracking_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['frame_index', 'image', 'frame_humans', 'frame_cars', 'unique_human_tracks_so_far'],
        )
        w.writeheader()
        w.writerows(csv_rows)

    print('Done. Saved tracking outputs to', out_dir)


def run_on_video(args, model, human_ids, car_ids):
    src = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print('Could not open video', src)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_dir / 'tracked.mp4'), fourcc, fps, (w, h))

    tracker_cfg = 'bytetrack.yaml' if args.tracker == 'bytetrack' else 'botsort.yaml'
    unique_human_tracks = set()
    csv_rows = []
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1

        result = model.track(
            source=frame,
            persist=True,
            tracker=tracker_cfg,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        plotted = result.plot()

        frame_humans = 0
        frame_cars = 0
        human_track_ids = []

        if result.boxes is not None and len(result.boxes) > 0:
            classes = result.boxes.cls.cpu().numpy().astype(int)
            ids = None
            if result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)

            for j, cls_id in enumerate(classes):
                if cls_id in human_ids:
                    frame_humans += 1
                    if ids is not None:
                        human_track_ids.append(int(ids[j]))
                elif cls_id in car_ids:
                    frame_cars += 1

        for tid in human_track_ids:
            unique_human_tracks.add(tid)

        plotted = draw_counts(plotted, frame_humans, frame_cars, len(unique_human_tracks))
        writer.write(plotted)

        csv_rows.append(
            {
                'frame_index': idx,
                'image': f'frame_{idx:06d}',
                'frame_humans': frame_humans,
                'frame_cars': frame_cars,
                'unique_human_tracks_so_far': len(unique_human_tracks),
            }
        )

        if args.max > 0 and idx >= args.max:
            break

    cap.release()
    writer.release()

    with (out_dir / 'tracking_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(
            f,
            fieldnames=['frame_index', 'image', 'frame_humans', 'frame_cars', 'unique_human_tracks_so_far'],
        )
        w.writeheader()
        w.writerows(csv_rows)

    print('Done. Saved tracking outputs to', out_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', required=True, help='image folder or video path')
    p.add_argument('--weights', default='runs/detect/train-4/weights/best.pt')
    p.add_argument('--out', default='outputs/tracking_results')
    p.add_argument('--tracker', default='bytetrack', choices=['bytetrack', 'botsort'])
    p.add_argument('--device', default='cpu', help='inference device, for example cpu or 0')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--fps', type=float, default=12.0)
    p.add_argument('--max', type=int, default=300, help='max frames/images to process, <=0 means all')
    p.add_argument('--save-video', action='store_true', help='save output video for image-sequence mode')
    p.add_argument('--human-classes', default=None, help='comma separated class names to treat as humans')
    p.add_argument('--car-classes', default=None, help='comma separated class names to treat as cars')
    args = p.parse_args()

    model = YOLO(args.weights)
    names = model.model.names if hasattr(model, 'model') and hasattr(model.model, 'names') else {0: 'person', 2: 'car'}

    human_ids, car_ids = resolve_target_classes(
        names,
        parse_class_list(args.human_classes),
        parse_class_list(args.car_classes),
    )
    print(f'Using class ids for humans: {sorted(human_ids)} | cars: {sorted(car_ids)}')

    src = Path(args.src)
    if src.is_dir():
        run_on_images(args, model, human_ids, car_ids)
    else:
        run_on_video(args, model, human_ids, car_ids)


if __name__ == '__main__':
    main()
