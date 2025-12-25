import numpy as np

def estimate_distance_px(box_height, focal_length_px=800):
    """
    Rough estimate: distance ≈ f / h
    """
    if box_height <= 0:
        return None
    return focal_length_px / box_height


def relative_speed(prev_h, curr_h, fps):
    """
    Approximate change in height over time.
    If height increases → object approaching.
    """
    return (curr_h - prev_h) * fps


def time_to_collision(prev_h, curr_h, fps, focal_length_px=800):
    d_prev = estimate_distance_px(prev_h, focal_length_px)
    d_curr = estimate_distance_px(curr_h, focal_length_px)

    if d_prev is None or d_curr is None:
        return None

    v_rel = relative_speed(prev_h, curr_h, fps)

    if v_rel <= 0:
        return None  # not approaching

    return d_curr / v_rel  # seconds
