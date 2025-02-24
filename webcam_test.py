import cv2
from ultralytics import YOLO

# Load ONNX model
model_path = "yolov8l_bdd10k.onnx"  
model = YOLO(model_path)

# Open webcam (or replace with video path)
video_source = 0  
cap = cv2.VideoCapture(video_source)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit if the video ends
    
    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Run inference
    results = model(frame)

    for result in results:
        for i, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box[:4])  # Convert to int

            # Get class label correctly
            class_id = int(result.boxes.cls[i])  # Class index
            label = result.names[class_id] if class_id in result.names else "Unknown"
            conf = result.boxes.conf[i]  # Confidence

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output
    cv2.imshow("YOLO ONNX - Real-Time", frame)

    if cv2.waitKey(1) & 0xFF == ord(" "):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
