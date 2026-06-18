from ultralytics import YOLO
import cv2
import time

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(
    frame,
    conf=0.6,
    imgsz=1280
)

    annotated_frame = results[0].plot()

    curr_time = time.time()

    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("CAMDEC", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()