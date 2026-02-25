import time
import math
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
latest_shoulders_px = None  # (left_x, left_y, right_x, right_y, dist_px)

# Pose landmark indices (MediaPipe Pose)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

def result_callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame_bgr, latest_shoulders_px

    frame_rgb = output_image.numpy_view()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = frame_bgr.shape

    latest_shoulders_px = None

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        ls = lm[LEFT_SHOULDER]
        rs = lm[RIGHT_SHOULDER]

        lx, ly = int(ls.x * w), int(ls.y * h)
        rx, ry = int(rs.x * w), int(rs.y * h)

        dist_px = math.hypot(rx - lx, ry - ly)
        latest_shoulders_px = (lx, ly, rx, ry, dist_px)

        # draw points + line
        cv2.circle(frame_bgr, (lx, ly), 6, (0, 255, 0), -1)
        cv2.circle(frame_bgr, (rx, ry), 6, (0, 255, 0), -1)
        cv2.line(frame_bgr, (lx, ly), (rx, ry), (0, 255, 0), 2)

        cv2.putText(
            frame_bgr,
            f"Shoulder width: {dist_px:.1f} px",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

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
        landmarker.detect_async(mp_image, int(time.time() * 1000))

        if latest_frame_bgr is not None:
            cv2.imshow("Shoulder Pixels (press Q)", latest_frame_bgr)
        else:
            cv2.imshow("Shoulder Pixels (press Q)", frame)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()