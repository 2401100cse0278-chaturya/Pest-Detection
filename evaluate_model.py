from ultralytics import YOLO

model = YOLO(
    "runs/detect/runs/pest_detection/experiment/weights/best.pt"
)

print("Evaluating Pest Detection Model...")
metrics = model.val(
    data="data.yaml",
    split="val",
    imgsz=640,
    batch=4,
    plots=True
)
print("\n===== MODEL EVALUATION RESULTS =====")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP@50: {metrics.box.map50:.4f}")
print(f"mAP@50-95: {metrics.box.map:.4f}")

print("\nEvaluation completed!")