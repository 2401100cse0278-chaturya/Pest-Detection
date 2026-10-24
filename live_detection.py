from ultralytics import YOLO
import cv2

model = YOLO(
    "runs/detect/runs/pest_detection/experiment/weights/best.pt"
)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    print("Check camera permissions and close other apps using it.")
    exit()

print("Live pest detection started!")
print("Press Q to stop.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read from camera.")
        break

    
    results = model.predict(
        source=frame,
        conf=0.25,
        verbose=False
    )

    
    annotated_frame = results[0].plot()

   
    cv2.imshow("Live Pest Detection", annotated_frame)

 
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()