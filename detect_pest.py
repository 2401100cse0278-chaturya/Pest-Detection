from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/pest_detection/experiment/weights/best.pt"
)
results = model.predict(
    source="new_pest.jpeg",
    conf=0.25,
    save=True
)

print("Pest detection completed!")