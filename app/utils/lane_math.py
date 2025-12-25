import numpy as np
import cv2

# -------------------------
# Extract lane pixels
# -------------------------
def get_lane_points(mask):
    """
    mask: binary lane mask (0/255)
    returns xs, ys, (w,h)
    """
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    return xs, ys, w, h


# -------------------------
# Fit polynomial x = Ay^2 + By + C
# -------------------------
def fit_lane_poly(xs, ys, h):
    if len(xs) < 80:   # too few points
        return None, None, None

    fit = np.polyfit(ys, xs, 2)
    ploty = np.linspace(0, h-1, h)
    fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
    return fit, ploty, fitx


# -------------------------
# Curvature calculation
# -------------------------
def lane_curvature(fit, y_eval):
    A, B = fit[0], fit[1]
    curvature = ((1 + (2*A*y_eval + B)**2)**1.5) / abs(2*A + 1e-6)
    return curvature


# -------------------------
# Lateral offset from lane center
# -------------------------
def lateral_offset_pixels(fit, w, h):
    yb = h - 1  # bottom row
    lane_bottom_x = fit[0]*yb**2 + fit[1]*yb + fit[2]
    vehicle_center = w / 2

    # +ve => lane is right side, -ve => left side
    return vehicle_center - lane_bottom_x


# -------------------------
# Drift speed based on lane shift across frames
# -------------------------
def drift_speed(prev_offset, curr_offset, fps):
    return (curr_offset - prev_offset) * fps  # px/sec


# -------------------------
# Time-to-lane-departure
# -------------------------
def time_to_lane_departure(offset_px, drift_speed_px_s, lane_width_px=350):
    """
    Positive offset: car is right of lane center
    Negative offset: car is left of lane center
    """
    if drift_speed_px_s == 0:
        return None

    if offset_px > 0:
        distance_remaining = abs(lane_width_px - offset_px)
    else:
        distance_remaining = abs(lane_width_px - abs(offset_px))

    if drift_speed_px_s <= 0:
        return None  # drifting back to center

    return distance_remaining / drift_speed_px_s
