from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(task="detect", data="data.yaml", epochs=500, patience=0, batch=1, imgsz=640, device=0, pretrained=False, optimizer="AdamW", resume=False, val=True)