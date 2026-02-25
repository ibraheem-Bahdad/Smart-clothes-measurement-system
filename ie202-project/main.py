import ctypes
import os
import platform
import time
from importlib import resources
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from calibration_process import (
    CalibrationParams,
    append_calibration_sample,
    build_device_calibration,
    calibrate_step,
    load_device_calibration,
    new_calib_samples,
    save_device_calibration,
)
from calculations import (
    compute_chest_and_length_geometry,
    get_point,
    landmark_conf,
    measurements_in_inches,
    median_of_samples,
    to_int_pt,
)
from data_sheet_decision import best_size

# ---------------- CONFIG ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"
WINDOW_NAME = "Smart Mirror - T-Shirt Size"

MEASURE_SECONDS = 5.0
MIN_CAPTURE_SAMPLES = 25
CALIB_SECONDS = 5.0
MIN_CALIB_SAMPLES = 30
CALIB_FEATURE_TOL_IN = 0.0
CALIB_SCALE_SPREAD_TOL = 0.14
LINE_EMA_ALPHA = 0.82
CALIB_MAX_BUFFER_SAMPLES = 500

VISIBILITY_THRESHOLD = 0.40
HIP_VISIBILITY_THRESHOLD = 0.45

# One-time calibration body (your measurements)
CALIB_BODY_SHOULDER_IN = 16.0
CALIB_BODY_CHEST_IN = 18.5
CALIB_BODY_LENGTH_IN = 26.0
CHEST_CIRC_FROM_WIDTH_FACTOR = 2.0  # circumference ~= 2 x front chest width

# Upper-body-only geometry
CHEST_LEVEL_T = 0.23  # shoulder->hip interpolation for upper-chest line
LENGTH_FROM_SHOULDER_RATIO = CALIB_BODY_LENGTH_IN / CALIB_BODY_SHOULDER_IN

# Device lock tolerances (for multi-user use on same setup)
DIST_RATIO_MIN = 0.70
DIST_RATIO_MAX = 1.45
CENTER_X_TOL = 0.12
SLOPE_TOL = 0.18
# Strict alignment thresholds used after pressing M.
ALIGN_SHOULDER_TOL_IN = 0.3
ALIGN_CENTER_X_TOL = 0.02
ALIGN_SLOPE_TOL = 0.05

CALIBRATION_PATH = BASE_DIR / "device_calibration.json"
BTN_W = 220
BTN_H = 42
BTN_MARGIN = 16

# Pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP = 23, 24

CALIB_STEPS = ("shoulder", "chest", "length")
CALIB_STEP_TEXT = {
    "shoulder": "Step 1/3 Shoulder",
    "chest": "Step 2/3 Chest",
    "length": "Step 3/3 Length",
}
CALIB_STEP_HINT = {
    "shoulder": "Hold shoulders level",
    "chest": "Keep chest line visible and stay still",
    "length": "Keep shoulder-to-hip area visible",
}

CALIB_PARAMS = CalibrationParams(
    min_calib_samples=MIN_CALIB_SAMPLES,
    calib_feature_tol_in=CALIB_FEATURE_TOL_IN,
    calib_scale_spread_tol=CALIB_SCALE_SPREAD_TOL,
    calib_body_shoulder_in=CALIB_BODY_SHOULDER_IN,
    calib_body_chest_in=CALIB_BODY_CHEST_IN,
    calib_body_length_in=CALIB_BODY_LENGTH_IN,
)


# Runtime state (set in run)
device_cal = None
latest_frame = None
latest_live_measure = None
latest_ready = False
latest_ready_hint = "Not calibrated"
capture_samples = []
calib_samples = new_calib_samples()
reset_btn_rect = (0, 0, 0, 0)
reset_clicked = False
calib_status_msg = ""
calib_attempt = 0
state = "NEED_CALIB"  # NEED_CALIB, CALIBRATING, READY, CAPTURING, LOCKED
start_time = None
calib_start_time = None
locked_values = None
locked_size = None
smoothed_chest_y = None
smoothed_chest_px = None
smoothed_length_px = None
calib_step_index = 0
calib_step_results = {}
latest_alignment_ratio = None


def apply_mediapipe_windows_compat():
    """Work around MediaPipe Windows builds missing libmediapipe.free()."""
    if os.name != "nt":
        return

    from mediapipe.tasks.python.core import mediapipe_c_bindings

    if getattr(mediapipe_c_bindings, "_smart_mirror_compat_patched", False):
        return

    def load_raw_library_compat(signatures=()):
        if mediapipe_c_bindings._shared_lib is None:
            if os.name == "posix":
                if platform.system() == "Darwin":
                    lib_filename = "libmediapipe.dylib"
                else:
                    lib_filename = "libmediapipe.so"
            else:
                lib_filename = "libmediapipe.dll"
            lib_path_context = resources.files("mediapipe.tasks.c")
            absolute_lib_path = str(lib_path_context / lib_filename)
            mediapipe_c_bindings._shared_lib = ctypes.CDLL(absolute_lib_path)

        for signature in signatures:
            c_func = getattr(mediapipe_c_bindings._shared_lib, signature.func_name)
            c_func.argtypes = signature.argtypes
            c_func.restype = signature.restype

        try:
            mediapipe_c_bindings._shared_lib.free.argtypes = [ctypes.c_void_p]
            mediapipe_c_bindings._shared_lib.free.restype = None
        except AttributeError:
            msvcrt = ctypes.CDLL("msvcrt.dll")
            mediapipe_c_bindings._shared_lib.free = msvcrt.free
            mediapipe_c_bindings._shared_lib.free.argtypes = [ctypes.c_void_p]
            mediapipe_c_bindings._shared_lib.free.restype = None

        return mediapipe_c_bindings._shared_lib

    mediapipe_c_bindings.load_raw_library = load_raw_library_compat
    mediapipe_c_bindings._smart_mirror_compat_patched = True


def on_mouse(event, x, y, flags, param):
    global reset_clicked
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    x1, y1, x2, y2 = reset_btn_rect
    if x1 <= x <= x2 and y1 <= y <= y2:
        reset_clicked = True


def start_calibration():
    global device_cal, state, calib_samples, calib_start_time
    global capture_samples, locked_values, locked_size, calib_status_msg, calib_attempt
    global smoothed_chest_y, smoothed_chest_px, smoothed_length_px
    global calib_step_index, calib_step_results
    device_cal = None
    state = "CALIBRATING"
    calib_samples = new_calib_samples()
    calib_step_index = 0
    calib_step_results = {}
    capture_samples.clear()
    locked_values = None
    locked_size = None
    calib_start_time = time.time()
    calib_attempt = 0
    calib_status_msg = "Calibration started."
    smoothed_chest_y = None
    smoothed_chest_px = None
    smoothed_length_px = None


def trim_calibration_samples():
    global calib_samples
    for key in ("shoulder", "chest", "length"):
        n = len(calib_samples[key])
        if n > CALIB_MAX_BUFFER_SAMPLES:
            del calib_samples[key][0 : n - CALIB_MAX_BUFFER_SAMPLES]


def callback(result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame, latest_live_measure, latest_ready, latest_ready_hint
    global capture_samples, calib_samples, device_cal, state
    global smoothed_chest_y, smoothed_chest_px, smoothed_length_px
    global calib_step_index
    global latest_alignment_ratio

    frame_rgb = output_image.numpy_view()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = frame.shape

    latest_live_measure = None
    latest_ready = False
    latest_ready_hint = "Pose not found"

    if not result.pose_landmarks:
        smoothed_chest_y = None
        smoothed_chest_px = None
        smoothed_length_px = None
        latest_alignment_ratio = None
        latest_frame = frame
        return

    lm = result.pose_landmarks[0]
    if (
        landmark_conf(lm[L_SHOULDER]) < VISIBILITY_THRESHOLD
        or landmark_conf(lm[R_SHOULDER]) < VISIBILITY_THRESHOLD
    ):
        smoothed_chest_y = None
        smoothed_chest_px = None
        smoothed_length_px = None
        latest_alignment_ratio = None
        latest_ready_hint = "Shoulders not visible"
        latest_frame = frame
        return

    ls = get_point(lm, L_SHOULDER, w, h)
    rs = get_point(lm, R_SHOULDER, w, h)

    shoulder_y = (ls[1] + rs[1]) / 2.0
    shoulder_left = (ls[0], shoulder_y)
    shoulder_right = (rs[0], shoulder_y)
    shoulder_mid = ((ls[0] + rs[0]) / 2.0, shoulder_y)

    shoulder_px = abs(rs[0] - ls[0])
    if shoulder_px < 5.0:
        latest_ready_hint = "Too far from camera"
        latest_frame = frame
        return

    slope = (rs[1] - ls[1]) / max(abs(rs[0] - ls[0]), 1.0)
    center_x = shoulder_mid[0] / float(w)

    lh = None
    rh = None
    torso_dy_px = None
    hips_visible = (
        landmark_conf(lm[L_HIP]) >= HIP_VISIBILITY_THRESHOLD
        and landmark_conf(lm[R_HIP]) >= HIP_VISIBILITY_THRESHOLD
    )
    if hips_visible:
        lh = get_point(lm, L_HIP, w, h)
        rh = get_point(lm, R_HIP, w, h)
        torso_dy_px = ((lh[1] + rh[1]) / 2.0) - shoulder_y

    # Draw shoulder line.
    cv2.circle(frame, to_int_pt(ls), 6, (0, 255, 0), -1)
    cv2.circle(frame, to_int_pt(rs), 6, (0, 255, 0), -1)
    cv2.line(frame, to_int_pt(shoulder_left), to_int_pt(shoulder_right), (0, 255, 0), 2)

    length_from_torso_ratio = device_cal["length_from_torso_ratio"] if device_cal is not None else 1.0
    chest_left, chest_right, hem_mid, chest_px, length_px = compute_chest_and_length_geometry(
        shoulder_y=shoulder_y,
        shoulder_px=shoulder_px,
        shoulder_mid=shoulder_mid,
        ls=ls,
        rs=rs,
        lh=lh,
        rh=rh,
        hips_visible=hips_visible,
        torso_dy_px=torso_dy_px,
        chest_level_t=CHEST_LEVEL_T,
        calib_body_chest_in=CALIB_BODY_CHEST_IN,
        calib_body_shoulder_in=CALIB_BODY_SHOULDER_IN,
        length_from_shoulder_ratio=LENGTH_FROM_SHOULDER_RATIO,
        length_from_torso_ratio=length_from_torso_ratio,
    )

    # Smooth visual/measurement guide lines to reduce landmark jitter.
    chest_y_raw = chest_left[1]
    chest_px_raw = max(1.0, chest_right[0] - chest_left[0])
    length_px_raw = max(1.0, hem_mid[1] - shoulder_mid[1])
    if smoothed_chest_y is None:
        smoothed_chest_y = chest_y_raw
        smoothed_chest_px = chest_px_raw
        smoothed_length_px = length_px_raw
    else:
        one_minus_a = 1.0 - LINE_EMA_ALPHA
        smoothed_chest_y = LINE_EMA_ALPHA * smoothed_chest_y + one_minus_a * chest_y_raw
        smoothed_chest_px = LINE_EMA_ALPHA * smoothed_chest_px + one_minus_a * chest_px_raw
        smoothed_length_px = LINE_EMA_ALPHA * smoothed_length_px + one_minus_a * length_px_raw

    chest_px = smoothed_chest_px
    length_px = smoothed_length_px
    chest_left = (shoulder_mid[0] - chest_px * 0.5, smoothed_chest_y)
    chest_right = (shoulder_mid[0] + chest_px * 0.5, smoothed_chest_y)
    hem_mid = (shoulder_mid[0], shoulder_mid[1] + length_px)

    if state == "CALIBRATING":
        step_name = CALIB_STEPS[calib_step_index]
        append_calibration_sample(
            samples=calib_samples,
            step_name=step_name,
            shoulder_px=shoulder_px,
            slope=slope,
            center_x=center_x,
            chest_px=chest_px,
            length_px=length_px,
            torso_dy_px=torso_dy_px,
        )
        trim_calibration_samples()

    cv2.circle(frame, to_int_pt(chest_left), 5, (0, 200, 255), -1)
    cv2.circle(frame, to_int_pt(chest_right), 5, (0, 200, 255), -1)
    cv2.line(frame, to_int_pt(chest_left), to_int_pt(chest_right), (0, 200, 255), 2)
    cv2.line(frame, to_int_pt(shoulder_mid), to_int_pt(hem_mid), (255, 255, 0), 2)

    if device_cal is None:
        latest_ready_hint = "Device not calibrated"
        latest_frame = frame
        return

    # Device lock checks for all users.
    ratio = shoulder_px / device_cal["target_shoulder_px"]
    latest_alignment_ratio = ratio
    if state == "ALIGNING":
        align_ratio_min = max(0.01, (CALIB_BODY_SHOULDER_IN - ALIGN_SHOULDER_TOL_IN) / CALIB_BODY_SHOULDER_IN)
        align_ratio_max = (CALIB_BODY_SHOULDER_IN + ALIGN_SHOULDER_TOL_IN) / CALIB_BODY_SHOULDER_IN
        dist_ok = align_ratio_min <= ratio <= align_ratio_max
        center_ok = abs(center_x - device_cal["target_center_x"]) <= ALIGN_CENTER_X_TOL
        angle_ok = abs(slope - device_cal["target_slope"]) <= ALIGN_SLOPE_TOL
    else:
        dist_ok = DIST_RATIO_MIN <= ratio <= DIST_RATIO_MAX
        center_ok = abs(center_x - device_cal["target_center_x"]) <= CENTER_X_TOL
        angle_ok = abs(slope - device_cal["target_slope"]) <= SLOPE_TOL

    if not dist_ok:
        if ratio < (align_ratio_min if state == "ALIGNING" else DIST_RATIO_MIN):
            latest_ready_hint = "Move closer"
        else:
            latest_ready_hint = "Move farther"
    elif not center_ok:
        latest_ready_hint = "Move to center"
    elif not angle_ok:
        latest_ready_hint = "Face straight"
    else:
        latest_ready = True
        latest_ready_hint = "Stop. Hold still"

    inch_per_px_shoulder = device_cal["inch_per_px_shoulder"]
    inch_per_px_chest = device_cal["inch_per_px_chest"]
    inch_per_px_length = device_cal["inch_per_px_length"]
    shoulder_in, chest_in, chest_circ_in, length_in = measurements_in_inches(
        shoulder_px=shoulder_px,
        chest_px=chest_px,
        length_px=length_px,
        inch_per_px_shoulder=inch_per_px_shoulder,
        inch_per_px_chest=inch_per_px_chest,
        inch_per_px_length=inch_per_px_length,
        chest_circ_factor=CHEST_CIRC_FROM_WIDTH_FACTOR,
    )

    sane = 12.0 <= shoulder_in <= 24.0 and 10.0 <= chest_in <= 24.0 and 16.0 <= length_in <= 30.0
    if sane and latest_ready:
        latest_live_measure = {
            "shoulder": shoulder_in,
            "chest": chest_in,
            "chest_circ": chest_circ_in,
            "length": length_in,
        }
        if state == "CAPTURING":
            capture_samples.append(latest_live_measure.copy())

    if state not in ("LOCKED", "ALIGNING", "CAPTURING"):
        cv2.putText(
            frame,
            (
                f"Scale S/C/L: "
                f"{device_cal['inch_per_px_shoulder']:.4f}/"
                f"{device_cal['inch_per_px_chest']:.4f}/"
                f"{device_cal['inch_per_px_length']:.4f}"
            ),
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 200, 100),
            2,
        )
        hint_color = (0, 255, 0) if latest_ready else (30, 30, 255)
        cv2.putText(
            frame,
            latest_ready_hint,
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            hint_color,
            2,
        )
        if latest_live_measure is not None:
            cv2.putText(
                frame,
                f"Shoulder width: {latest_live_measure['shoulder']:.2f} in",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Chest width: {latest_live_measure['chest']:.2f} in",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 200, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Chest circumference: {latest_live_measure['chest_circ']:.2f} in",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (0, 165, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Upper length: {latest_live_measure['length']:.2f} in",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 0),
                2,
            )

    latest_frame = frame


def run():
    global device_cal, latest_frame, latest_live_measure, latest_ready, latest_ready_hint
    global capture_samples, calib_samples, reset_btn_rect, reset_clicked
    global calib_status_msg, calib_attempt, state, start_time, calib_start_time
    global locked_values, locked_size
    global smoothed_chest_y, smoothed_chest_px, smoothed_length_px
    global calib_step_index, calib_step_results
    global latest_alignment_ratio

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Unable to open file at {MODEL_PATH}")

    apply_mediapipe_windows_compat()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")

    device_cal = load_device_calibration(CALIBRATION_PATH, CALIB_PARAMS)
    if device_cal is None:
        print("No previous device calibration found. Click Calibrate button.")
    else:
        print("Loaded previous device calibration.")

    latest_frame = None
    latest_live_measure = None
    latest_ready = False
    latest_ready_hint = "Not calibrated"
    capture_samples = []
    calib_samples = new_calib_samples()
    reset_btn_rect = (0, 0, 0, 0)
    reset_clicked = False
    calib_status_msg = ""
    calib_attempt = 0

    state = "READY" if device_cal is not None else "NEED_CALIB"  # NEED_CALIB, CALIBRATING, READY, ALIGNING, CAPTURING, LOCKED
    start_time = None
    calib_start_time = None
    locked_values = None
    locked_size = None
    smoothed_chest_y = None
    smoothed_chest_px = None
    smoothed_length_px = None
    calib_step_index = 0
    calib_step_results = {}
    latest_alignment_ratio = None

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        result_callback=callback,
    )

    landmarker = vision.PoseLandmarker.create_from_options(options)

    try:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse)

        while True:
            now = time.time()
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_image, int(now * 1000))

            display = latest_frame if latest_frame is not None else frame
            h_disp, w_disp = display.shape[:2]
            x2 = w_disp - BTN_MARGIN
            y1 = BTN_MARGIN
            x1 = max(0, x2 - BTN_W)
            y2 = y1 + BTN_H
            reset_btn_rect = (x1, y1, x2, y2)

            cv2.rectangle(display, (x1, y1), (x2, y2), (40, 40, 40), -1)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                display,
                "Calibrate",
                (x1 + 52, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2,
            )

            if reset_clicked:
                start_calibration()
                reset_clicked = False

            if state in ("NEED_CALIB", "READY"):
                if state == "NEED_CALIB":
                    top_msg = "Click Calibrate button"
                else:
                    top_msg = "Press M to start locked alignment"
                cv2.putText(
                    display,
                    top_msg,
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                )

            elif state == "CALIBRATING":
                if calib_start_time is None:
                    calib_start_time = now
                elapsed = now - calib_start_time
                remaining = max(0.0, CALIB_SECONDS - elapsed)
                step_name = CALIB_STEPS[calib_step_index]
                step_title = CALIB_STEP_TEXT[step_name]
                step_hint = CALIB_STEP_HINT[step_name]
                step_n = len(calib_samples[step_name])
                shoulder_n = len(calib_samples["shoulder"])
                chest_n = len(calib_samples["chest"])
                length_n = len(calib_samples["length"])

                cv2.putText(
                    display,
                    f"{step_title}... {remaining:.1f}s",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    step_hint,
                    (20, 74),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    calib_status_msg,
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Step samples: {step_n}/{MIN_CALIB_SAMPLES}",
                    (20, 104),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"All S/C/L: {shoulder_n}/{chest_n}/{length_n}",
                    (20, 132),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                )
                if remaining <= 0:
                    calib_attempt += 1
                    step_result, fail_reason = calibrate_step(calib_samples, step_name, CALIB_PARAMS)
                    if step_result is not None:
                        calib_step_results[step_name] = step_result
                        calib_status_msg = f"{step_title} done: {step_result['inch_per_px']:.4f} in/px"
                        if calib_step_index < len(CALIB_STEPS) - 1:
                            calib_step_index += 1
                            next_step = CALIB_STEPS[calib_step_index]
                            calib_samples[next_step] = []
                            calib_start_time = now
                        else:
                            cal = build_device_calibration(calib_step_results, CALIB_PARAMS)
                            device_cal = cal
                            save_device_calibration(CALIBRATION_PATH, cal)
                            state = "READY"
                            calib_start_time = None
                            calib_status_msg = (
                                "Calibration complete S/C/L="
                                f"{cal['inch_per_px_shoulder']:.4f}/"
                                f"{cal['inch_per_px_chest']:.4f}/"
                                f"{cal['inch_per_px_length']:.4f}"
                            )
                    else:
                        calib_status_msg = f"{step_title} attempt #{calib_attempt}: {fail_reason}"
                        calib_samples[step_name] = []
                        calib_start_time = now

            elif state == "ALIGNING":
                cv2.putText(
                    display,
                    "LOCKED ALIGNMENT",
                    (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    latest_ready_hint,
                    (20, 74),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.62,
                    (255, 255, 255),
                    2,
                )
                if latest_alignment_ratio is not None:
                    cv2.putText(
                        display,
                        f"Distance ratio: {latest_alignment_ratio:.3f} (target 1.000)",
                        (20, 104),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.56,
                        (255, 255, 255),
                        2,
                    )
                if latest_ready:
                    capture_samples.clear()
                    start_time = time.time()
                    state = "CAPTURING"

            elif state == "CAPTURING":
                remaining = max(0.0, MEASURE_SECONDS - (now - start_time))
                cv2.putText(
                    display,
                    "STOP - measuring now",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Keep still... {remaining:0.1f}s",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )

                if remaining <= 0:
                    if len(capture_samples) >= MIN_CAPTURE_SAMPLES:
                        measured = {
                            "shoulder": median_of_samples(capture_samples, "shoulder"),
                            "chest": median_of_samples(capture_samples, "chest"),
                            "chest_circ": median_of_samples(capture_samples, "chest_circ"),
                            "length": median_of_samples(capture_samples, "length"),
                        }
                        locked_values = measured
                        locked_size = best_size(measured)
                        state = "LOCKED"
                    else:
                        state = "READY"
                    capture_samples.clear()

            elif state == "LOCKED":
                cv2.putText(
                    display,
                    "RESULT",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    display,
                    f"Shoulder width: {locked_values['shoulder']:.2f} in",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Chest width: {locked_values['chest']:.2f} in",
                    (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Chest circumference: {locked_values['chest_circ']:.2f} in",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Upper length: {locked_values['length']:.2f} in",
                    (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display,
                    f"Recommended size: {locked_size}",
                    (20, 205),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.95,
                    (0, 255, 0),
                    3,
                )
                cv2.putText(
                    display,
                    "Press M measure again or click Calibrate",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("m"), ord("M")) and state in ("READY", "LOCKED"):
                latest_live_measure = None
                state = "ALIGNING"
            if key in (ord("c"), ord("C")):
                start_calibration()

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    run()
