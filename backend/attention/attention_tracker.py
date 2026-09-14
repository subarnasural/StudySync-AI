import cv2
import time

latest_frame_bytes = None


def gen_frames(is_active_callback):
    global latest_frame_bytes
    import numpy as np
    
    # Pre-render a loading placeholder so the browser establishes a valid stream immediately
    placeholder = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Connecting to camera...", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    _, jpeg = cv2.imencode('.jpg', placeholder)
    placeholder_bytes = jpeg.tobytes()

    while is_active_callback():
        frame_to_yield = latest_frame_bytes if latest_frame_bytes is not None else placeholder_bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
        time.sleep(0.04)



class AttentionTracker:

    def __init__(self):

        self.total_frames = 0
        self.attentive_frames = 0
        self.distracted_frames = 0

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def calculate_attention(
        self,
        is_active_callback=None,
        update_callback=None
    ):
        global latest_frame_bytes
        import logging
        logger = logging.getLogger(__name__)

        cap = None
        for attempt in range(1, 4):
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                break
            logger.warning("Webcam is busy or locked. Retrying in 1.5 seconds (Attempt %d/3)...", attempt)
            time.sleep(1.5)

        if not cap or not cap.isOpened():
            raise Exception("Could not open webcam after 3 attempts. It may be in use by another application.")

        results = {
            "total_frames": 0,
            "attentive_frames": 0,
            "distracted_frames": 0,
            "attention_score": 0
        }

        while is_active_callback is None or is_active_callback():

            success, frame = cap.read()

            if not success:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(60, 60)
            )

            self.total_frames += 1

            attentive = len(faces) > 0

            if attentive:
                self.attentive_frames += 1
            else:
                self.distracted_frames += 1

            # Draw green bounding boxes around faces for premium live visual feedback
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Encode and cache the latest processed frame
            _, jpeg = cv2.imencode('.jpg', frame)
            latest_frame_bytes = jpeg.tobytes()

            attention_score = (
                self.attentive_frames / self.total_frames
            ) * 100

            results = {
                "total_frames": self.total_frames,
                "attentive_frames": self.attentive_frames,
                "distracted_frames": self.distracted_frames,
                "attention_score": round(attention_score, 2)
            }

            # IMPORTANT: SEND LIVE RESULTS
            if update_callback:
                update_callback(results)

            time.sleep(0.03)

        cap.release()

        latest_frame_bytes = None

        return results