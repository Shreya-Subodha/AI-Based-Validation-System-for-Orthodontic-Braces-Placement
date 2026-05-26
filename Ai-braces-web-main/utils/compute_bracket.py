import math


PLACEMENT_THRESHOLD_PX = 20   # pixels — adjust after calibration


def compute_error(ideal_point: tuple, detected_box: dict) -> dict:
    """
    Compute the distance between an ideal bracket point (from UNet)
    and the centre of a detected bracket bounding box (from YOLO).

    Args:
        ideal_point:   (x, y) of the ideal bracket position.
        detected_box:  dict with keys cx, cy (centre of YOLO bounding box).

    Returns:
        dict with keys:
            distance  – Euclidean distance in pixels (float)
            status    – "Correct Placement" or "Needs Adjustment"
    """
    dx = ideal_point[0] - detected_box["cx"]
    dy = ideal_point[1] - detected_box["cy"]
    distance = math.hypot(dx, dy)

    status = "Correct Placement" if distance <= PLACEMENT_THRESHOLD_PX else "Needs Adjustment"

    return {"distance": round(distance, 2), "status": status}


def match_brackets(ideal_points: list, detected_boxes: list) -> list:
    """
    Greedily match each detected bracket to its nearest ideal point.

    Returns a list of match dicts:
        ideal_point, detected_box, distance, status
    """
    if not ideal_points or not detected_boxes:
        return []

    results = []
    used_ideals = set()

    for box in detected_boxes:
        best_dist = float("inf")
        best_idx  = -1

        for i, pt in enumerate(ideal_points):
            if i in used_ideals:
                continue
            dx = pt[0] - box["cx"]
            dy = pt[1] - box["cy"]
            d  = math.hypot(dx, dy)
            if d < best_dist:
                best_dist = d
                best_idx  = i

        if best_idx >= 0:
            used_ideals.add(best_idx)
            error = compute_error(ideal_points[best_idx], box)
            results.append({
                "ideal_point":   ideal_points[best_idx],
                "detected_box":  box,
                "distance":      error["distance"],
                "status":        error["status"],
            })

    return results
