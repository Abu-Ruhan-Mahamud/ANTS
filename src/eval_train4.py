from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/detect/train-4/weights/best.pt')
    r = model.val(
        data='VisDrone_Dataset/visdrone.yaml',
        split='val',
        imgsz=640,
        batch=16,
        device='cpu',
        plots=True,
    )
    print('precision', r.box.mp)
    print('recall', r.box.mr)
    print('map50', r.box.map50)
    print('map50_95', r.box.map)
