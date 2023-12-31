from sys import argv
from ultralytics import YOLO

model = YOLO("./runs/detect/train/weights/best.pt")
result = model.predict(task="detect", source="test/images", conf=0.5, save=True, save_crop=True, device=0)
