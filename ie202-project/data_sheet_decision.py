SIZE_CHART_IN = {
    # Chart source (from your image):
    # Chest circumference XS/S/M/L = 32-34 / 35-37 / 38-40 / 41-43.
    # System chest is front width, so use midpoint(circumference)/2.
    "XS": {"chest": 16.5, "shoulder": 16.0, "length": 26.5},
    "S": {"chest": 18.0, "shoulder": 17.0, "length": 27.5},
    "M": {"chest": 19.5, "shoulder": 18.0, "length": 28.5},
    "L": {"chest": 21.0, "shoulder": 19.0, "length": 29.5},
}


def best_size(meas):
    best = None
    best_score = 1e18

    # Round live measurements before decision scoring to reduce jitter noise.
    target_chest = round(meas["chest"], 1)
    target_shoulder = round(meas["shoulder"], 1)
    target_length = round(meas["length"], 1)

    for size, vals in SIZE_CHART_IN.items():
        chest_diff = target_chest - vals["chest"]
        shoulder_diff = target_shoulder - vals["shoulder"]
        length_diff = target_length - vals["length"]
        score = abs(chest_diff) * 1.8 + abs(shoulder_diff) * 1.3 + abs(length_diff) * 1.0
        if chest_diff > 0:
            score += chest_diff * 1.2
        if shoulder_diff > 0:
            score += shoulder_diff * 1.0
        if length_diff > 0:
            score += length_diff * 0.3
        if score < best_score:
            best_score = score
            best = size
    return best
