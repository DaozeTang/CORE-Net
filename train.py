import warnings
warnings.filterwarnings('ignore', category=UserWarning)
from ultralytics import YOLO

if __name__ == "__main__":
    #model = YOLO("ultralytics/cfg/models/11/yolo11n.yaml")
    model = YOLO("ultralytics/cfg/models/CORE-Net/CORE-Net.yaml")
    model.train(
        data="ultralytics/cfg/datasets/LLVIP.yaml",
        #data="ultralytics/cfg/datasets/DroneVehicle.yaml",
        cache=False,
        imgsz=640,
        epochs=200,
        batch=16,
        close_mosaic=0,
        workers=128,
        device="0,1",
        optimizer="SGD",
        patience=0,
        amp=True,  # Bypass the AMP check during RGB+IR training or set to False.
        project="runs/train/LLVIP",
        #project="runs/train/DroneVehicle",
        #project="runs/train/DroneVehicle/Ablation",
        name="CORE-Net",
    )
