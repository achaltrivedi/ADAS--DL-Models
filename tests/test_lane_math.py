import numpy as np
from app.utils.lane_math import (
    fit_lane_poly, lateral_offset_pixels, drift_speed, time_to_lane_departure
)

def test_lane_math():
    # Fake lane boundary
    ys = np.linspace(0, 719, 720)
    xs = 640 + 20 * np.sin(ys / 50)  # small curve

    fit, ploty, fitx = fit_lane_poly(xs, ys, 720)
    assert fit is not None

    offset = lateral_offset_pixels(fit, 1280, 720)
    print("Offset:", offset)

    drift = drift_speed(prev_offset=offset-5, curr_offset=offset, fps=30)
    print("Drift px/s:", drift)

    tld = time_to_lane_departure(offset, drift)
    print("TLD:", tld)

    print("Lane math OK ✔")

if __name__ == "__main__":
    test_lane_math()
