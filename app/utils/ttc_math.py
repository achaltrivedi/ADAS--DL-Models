# utils/ttc_math.py
print("LOADED TTC FILE:", __file__)

import numpy as np

def estimate_distance_px(box_height, focal_length_px: float = 800.0):
    """
    Rough distance estimate from bounding-box height:
        distance ≈ f / h

    Returns:
        float distance in "pixel units" or None if not valid.
    """
    # Handle invalid input
    if box_height is None:
        return None

    # Sometimes you might get numpy scalars
    if isinstance(box_height, (np.ndarray,)):
        if box_height.size == 0:
            return None
        box_height = float(box_height)

    # Reject non-positive heights
    if box_height <= 0:
        return None

    return float(focal_length_px) / float(box_height)


def relative_speed(prev_h, curr_h, fps: float):
    """
    Approximate relative speed based on change in box height.
    If height increases → object appears closer (approaching).
    """
    return (curr_h - prev_h) * fps


def time_to_collision(prev_h, curr_h, fps: float, focal_length_px: float = 800.0):
    """
    Very rough TTC estimate using only bounding-box height:

      1. Convert previous and current heights to pseudo-distances.
      2. Compute relative speed from height change.
      3. TTC = current_distance / relative_speed

    Returns:
        TTC in seconds, or None if:
        - heights are invalid
        - object is not approaching (or stationary)
    """
    # Distance estimates
    d_prev = estimate_distance_px(prev_h, focal_length_px)
    d_curr = estimate_distance_px(curr_h, focal_length_px)

    # If either distance cannot be estimated → no TTC
    if d_prev is None or d_curr is None:
        return None

    # Relative speed (in these pseudo-units)
    v_rel = relative_speed(prev_h, curr_h, fps)

    # Not approaching (moving away or stationary) → no TTC
    if v_rel <= 0:
        return None

    return d_curr / v_rel  # seconds
