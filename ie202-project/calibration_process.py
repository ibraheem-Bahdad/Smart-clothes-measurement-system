import json
import os
import statistics
import time
from dataclasses import dataclass

from calculations import clamp


@dataclass(frozen=True)
class CalibrationParams:
    min_calib_samples: int
    calib_feature_tol_in: float
    calib_scale_spread_tol: float
    calib_body_shoulder_in: float
    calib_body_chest_in: float
    calib_body_length_in: float


def save_device_calibration(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_device_calibration(path, params):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required = [
            "target_shoulder_px",
            "target_center_x",
            "target_slope",
            "length_from_torso_ratio",
        ]
        for k in required:
            if k not in data:
                return None

        # If calibration reference values changed, force recalibration.
        ref_vals = {
            "calib_body_shoulder_in": params.calib_body_shoulder_in,
            "calib_body_chest_in": params.calib_body_chest_in,
            "calib_body_length_in": params.calib_body_length_in,
        }
        for k, v in ref_vals.items():
            if k in data and abs(float(data[k]) - float(v)) > 0.01:
                return None

        # Backward compatibility with single-scale calibration files.
        if "inch_per_px_shoulder" not in data:
            if "inch_per_px" in data:
                base = float(data["inch_per_px"])
            else:
                base = float(params.calib_body_shoulder_in) / float(data["target_shoulder_px"])
            data["inch_per_px_shoulder"] = base
            data["inch_per_px_chest"] = base
            data["inch_per_px_length"] = base

        if "target_chest_px" not in data:
            data["target_chest_px"] = params.calib_body_chest_in / float(data["inch_per_px_chest"])
        if "target_length_px" not in data:
            data["target_length_px"] = params.calib_body_length_in / float(data["inch_per_px_length"])

        data["inch_per_px"] = statistics.mean(
            [
                float(data["inch_per_px_shoulder"]),
                float(data["inch_per_px_chest"]),
                float(data["inch_per_px_length"]),
            ]
        )
        return data
    except Exception:
        return None


def new_calib_samples():
    return {"shoulder": [], "chest": [], "length": []}


def append_calibration_sample(samples, step_name, shoulder_px, slope, center_x, chest_px, length_px, torso_dy_px):
    if step_name == "shoulder":
        samples["shoulder"].append(
            {
                "shoulder_px": shoulder_px,
                "slope": slope,
                "center_x": center_x,
            }
        )
    elif step_name == "chest":
        if chest_px is not None:
            samples["chest"].append({"chest_px": chest_px})
    elif step_name == "length":
        if length_px is not None:
            samples["length"].append(
                {
                    "length_px": length_px,
                    "torso_dy_px": torso_dy_px,
                }
            )


def calibrate_step(samples, step_name, params):
    if step_name == "shoulder":
        step_samples = samples["shoulder"]
        if len(step_samples) < params.min_calib_samples:
            return None, "not enough shoulder samples"
        shoulder_px = statistics.median([s["shoulder_px"] for s in step_samples])
        if shoulder_px <= 1.0:
            return None, "invalid shoulder pixels"
        return {
            "inch_per_px": params.calib_body_shoulder_in / shoulder_px,
            "target_px": shoulder_px,
            "center_x": statistics.median([s["center_x"] for s in step_samples]),
            "slope": statistics.median([s["slope"] for s in step_samples]),
        }, ""

    if step_name == "chest":
        step_samples = samples["chest"]
        if len(step_samples) < params.min_calib_samples:
            return None, "not enough chest samples"
        chest_px = statistics.median([s["chest_px"] for s in step_samples])
        if chest_px <= 1.0:
            return None, "invalid chest pixels"
        return {
            "inch_per_px": params.calib_body_chest_in / chest_px,
            "target_px": chest_px,
        }, ""

    if step_name == "length":
        step_samples = samples["length"]
        if len(step_samples) < params.min_calib_samples:
            return None, "not enough length samples"
        length_px = statistics.median([s["length_px"] for s in step_samples])
        if length_px <= 1.0:
            return None, "invalid length pixels"
        torso_candidates = [s["torso_dy_px"] for s in step_samples if s["torso_dy_px"] is not None and s["torso_dy_px"] > 1.0]
        torso_dy_px = statistics.median(torso_candidates) if len(torso_candidates) >= 10 else length_px
        return {
            "inch_per_px": params.calib_body_length_in / length_px,
            "target_px": length_px,
            "torso_dy_px": torso_dy_px,
        }, ""

    return None, "unknown calibration step"


def build_device_calibration(step_results, params):
    shoulder_step = step_results["shoulder"]
    chest_step = step_results["chest"]
    length_step = step_results["length"]

    inch_per_px_shoulder = float(shoulder_step["inch_per_px"])
    inch_per_px_chest = float(chest_step["inch_per_px"])
    inch_per_px_length = float(length_step["inch_per_px"])

    target_shoulder_px = params.calib_body_shoulder_in / inch_per_px_shoulder
    target_chest_px = params.calib_body_chest_in / inch_per_px_chest
    target_length_px = params.calib_body_length_in / inch_per_px_length

    torso_dy_px = max(float(length_step["torso_dy_px"]), 1.0)
    length_from_torso_ratio = target_length_px / torso_dy_px
    length_from_torso_ratio = clamp(length_from_torso_ratio, 0.75, 1.20)

    return {
        "inch_per_px": float(statistics.mean([inch_per_px_shoulder, inch_per_px_chest, inch_per_px_length])),
        "inch_per_px_shoulder": inch_per_px_shoulder,
        "inch_per_px_chest": inch_per_px_chest,
        "inch_per_px_length": inch_per_px_length,
        "target_shoulder_px": float(target_shoulder_px),
        "target_chest_px": float(target_chest_px),
        "target_length_px": float(target_length_px),
        "target_center_x": float(shoulder_step["center_x"]),
        "target_slope": float(shoulder_step["slope"]),
        "length_from_torso_ratio": float(length_from_torso_ratio),
        "calib_shoulder_px": float(target_shoulder_px),
        "calib_chest_px": float(target_chest_px),
        "calib_length_px": float(target_length_px),
        "calib_body_shoulder_in": params.calib_body_shoulder_in,
        "calib_body_chest_in": params.calib_body_chest_in,
        "calib_body_length_in": params.calib_body_length_in,
        "created_at": int(time.time()),
    }
