from ultralytics import YOLO
from pathlib import Path
data_file = Path("data.yaml")

if not data_file.exists():
    raise FileNotFoundError("data.yaml not found in the project folder")

print("Loading YOLO model...")
model = YOLO("yolov8n.pt")

print("Starting pest detection model training...")

results = model.train(
    data=str(data_file.resolve()),
    epochs=30,
    imgsz=640,
    batch=4,
    project="runs/pest_detection",
    name="experiment",
    patience=5
)

print("Training completed!")
print("Check the runs/pest_detection folder for results.")