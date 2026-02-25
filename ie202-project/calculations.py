import statistics


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def to_int_pt(p):
    return (int(round(p[0])), int(round(p[1])))


def landmark_conf(lm):
    vals = []
    vis = getattr(lm, "visibility", None)
    pres = getattr(lm, "presence", None)
    if vis is not None:
        vals.append(float(vis))
    if pres is not None:
        vals.append(float(pres))
    if not vals:
        return 1.0
    return max(vals)


def get_point(lm, idx, w, h):
    return (lm[idx].x * w, lm[idx].y * h)


def median_of_samples(samples, key):
    return statistics.median([s[key] for s in samples])


def compute_calibration_features(
    ls,
    rs,
    lh,
    rh,
    hips_visible,
    torso_dy_px,
    shoulder_px,
    chest_level_t,
    chest_to_shoulder_ratio,
    length_from_shoulder_ratio,
):
    # Fallback ratios keep calibration progressing when hip landmarks are unstable.
    fallback_chest_px = shoulder_px * chest_to_shoulder_ratio
    fallback_length_px = shoulder_px * length_from_shoulder_ratio
    chest_px = fallback_chest_px
    length_px = fallback_length_px

    if hips_visible and torso_dy_px is not None and torso_dy_px > shoulder_px * 0.10:
        chest_t = chest_level_t
        left_x = ls[0] + (lh[0] - ls[0]) * chest_t
        right_x = rs[0] + (rh[0] - rs[0]) * chest_t
        chest_measured = abs(right_x - left_x)
        length_measured = torso_dy_px
        blend = 0.70
        chest_px = chest_measured * blend + fallback_chest_px * (1.0 - blend)
        length_px = length_measured * blend + fallback_length_px * (1.0 - blend)
    return chest_px, length_px


def compute_chest_and_length_geometry(
    shoulder_y,
    shoulder_px,
    shoulder_mid,
    ls,
    rs,
    lh,
    rh,
    hips_visible,
    torso_dy_px,
    chest_level_t,
    calib_body_chest_in,
    calib_body_shoulder_in,
    length_from_shoulder_ratio,
    length_from_torso_ratio,
):
    chest_y = shoulder_y + shoulder_px * 0.10
    chest_px_raw = shoulder_px * (calib_body_chest_in / calib_body_shoulder_in)
    length_px = shoulder_px * length_from_shoulder_ratio

    if hips_visible and torso_dy_px is not None and torso_dy_px > shoulder_px * 0.10:
        chest_y = shoulder_y + torso_dy_px * chest_level_t
        chest_t = chest_level_t
        left_x = ls[0] + (lh[0] - ls[0]) * chest_t
        right_x = rs[0] + (rh[0] - rs[0]) * chest_t
        chest_px_raw = abs(right_x - left_x)
        length_px = torso_dy_px * length_from_torso_ratio

    chest_px = clamp(chest_px_raw, shoulder_px * 0.70, shoulder_px * 1.35)
    length_px = clamp(length_px, shoulder_px * 0.95, shoulder_px * 1.70)

    chest_left = (shoulder_mid[0] - chest_px * 0.5, chest_y)
    chest_right = (shoulder_mid[0] + chest_px * 0.5, chest_y)
    hem_mid = (shoulder_mid[0], shoulder_mid[1] + length_px)
    return chest_left, chest_right, hem_mid, chest_px, length_px


def measurements_in_inches(
    shoulder_px,
    chest_px,
    length_px,
    inch_per_px_shoulder,
    inch_per_px_chest,
    inch_per_px_length,
    chest_circ_factor,
):
    shoulder_in = shoulder_px * inch_per_px_shoulder
    chest_in = chest_px * inch_per_px_chest
    chest_circ_in = chest_in * chest_circ_factor
    length_in = length_px * inch_per_px_length
    return shoulder_in, chest_in, chest_circ_in, length_in
