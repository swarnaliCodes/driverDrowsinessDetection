import cv2
import time
import threading
import argparse
from collections import deque
from ultralytics import YOLO
import platform
import os


# =========================
# Cross-Platform Alert Sound
# =========================
def play_alert():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 800)
    else:
        # Simple terminal bell for Linux/Mac
        print("\a")


# =========================
# Drowsiness Detection System
# =========================
class DrowsinessDetector:

    def __init__(self, model_path, camera_index=0):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Cannot open webcam")

        # Parameters
        self.class_names = ["eyeclosed", "yawn"]

        self.WINDOW_SIZE = 120
        self.FPS_ASSUMED = 30
        self.WINDOW_SECONDS = self.WINDOW_SIZE / self.FPS_ASSUMED

        self.W_PERCLOS = 0.7
        self.W_YAWN = 0.3
        self.DROWSY_SCORE_THRESHOLD = 0.5
        self.STABILITY_SECONDS = 2

        self.eye_window = deque(maxlen=self.WINDOW_SIZE)
        self.yawn_window = deque(maxlen=self.WINDOW_SIZE)

        self.alert_active = False
        self.drowsy_start_time = None

    def process_frame(self, frame):
        results = self.model(frame, conf=0.25, verbose=False)

        eye_closed = 0
        yawn = 0

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label = f"{self.class_names[cls_id]} {conf:.2f}"
                color = (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if self.class_names[cls_id] == "eyeclosed":
                    eye_closed = 1
                if self.class_names[cls_id] == "yawn":
                    yawn = 1

        return frame, eye_closed, yawn

    def compute_metrics(self):
        perclos = sum(self.eye_window) / self.WINDOW_SIZE
        yawns = sum(self.yawn_window)
        yawn_rate = yawns / self.WINDOW_SECONDS

        drowsy_score = (self.W_PERCLOS * perclos) + \
                       (self.W_YAWN * yawn_rate)

        return perclos, yawn_rate, drowsy_score

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            frame, eye_closed, yawn = self.process_frame(frame)

            self.eye_window.append(eye_closed)
            self.yawn_window.append(yawn)

            if len(self.eye_window) < self.WINDOW_SIZE:
                cv2.imshow("Driver Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            perclos, yawn_rate, drowsy_score = self.compute_metrics()

            current_time = time.time()

            if drowsy_score > self.DROWSY_SCORE_THRESHOLD:
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = current_time

                if current_time - self.drowsy_start_time >= self.STABILITY_SECONDS:
                    status = "DROWSY"
                    color = (0, 0, 255)

                    if not self.alert_active:
                        self.alert_active = True
                        threading.Thread(
                            target=play_alert, daemon=True).start()
                else:
                    status = "MONITORING"
                    color = (0, 255, 255)
            else:
                status = "ALERT"
                color = (0, 255, 0)
                self.alert_active = False
                self.drowsy_start_time = None

            cv2.putText(frame, f"Status: {status}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Yawn Rate: {yawn_rate:.2f}/s", (30, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.putText(frame, f"Score: {drowsy_score:.2f}", (30, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


# =========================
# Entry Point
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Real-Time Driver Drowsiness Detection")
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Path to trained YOLO model")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera index")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file not found at {args.model}")

    detector = DrowsinessDetector(
        model_path=args.model,
        camera_index=args.camera
    )
    detector.run()


if __name__ == "__main__":
    main()
