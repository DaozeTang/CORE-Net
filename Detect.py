from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/train/DroneVehicle/YOLO11n-RGB/weights/best.pt')
    model.predict(source='/root/autodl-fs/RGBIR/datasets/DroneVehicle/images/test',
                  imgsz=640,
                  project='runs/detect/DroneVehicle/test',
                  name='YOLO11n-RGB',
                  save=True,
                  show_labels=False,
                )