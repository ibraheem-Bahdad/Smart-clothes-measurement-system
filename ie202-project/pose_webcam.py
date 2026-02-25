import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera not opened. Try index 1 or close apps using the camera.")

latest_frame_bgr = None

def result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame_bgr
    frame_rgb = output_image.numpy_view()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if result.pose_landmarks:
        h, w, _ = frame_bgr.shape
        for lm in result.pose_landmarks[0]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame_bgr, (x, y), 4, (0, 255, 0), -1)

    latest_frame_bgr = frame_bgr

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.LIVE_STREAM,
    num_poses=1,
    result_callback=result_callback,
)

landmarker = PoseLandmarker.create_from_options(options)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.time() * 1000)

        landmarker.detect_async(mp_image, timestamp_ms)

        if latest_frame_bgr is not None:
            cv2.imshow("PoseLandmarker (new API) - press Q", latest_frame_bgr)
        else:
            cv2.imshow("PoseLandmarker (new API) - press Q", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()