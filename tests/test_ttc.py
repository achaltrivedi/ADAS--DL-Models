# tests/test_ttc.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_UTILS = ROOT / "app" / "utils"
sys.path.insert(0, str(APP_UTILS))

from app.utils.ttc_math import time_to_collision  # now correct


def test_ttc():
    fps = 30

    # 1) Object approaching (height increases)
    prev_h = 40
    curr_h = 60
    ttc = time_to_collision(prev_h, curr_h, fps)
    assert ttc is not None and ttc > 0, "TTC should be positive when object approaches."

    # 2) Object moving away (height decreases)
    prev_h = 60
    curr_h = 40
    ttc = time_to_collision(prev_h, curr_h, fps)
    assert ttc is None, "TTC should be None when object is moving away."

    # 3) No movement (same height)
    prev_h = 50
    curr_h = 50
    ttc = time_to_collision(prev_h, curr_h, fps)
    assert ttc is None, "TTC should be None when relative velocity is zero."

    # 4) Invalid input (None)
    ttc = time_to_collision(None, 40, fps)
    assert ttc is None, "TTC should be None for invalid height input."

    print("TTC test passed ✔")

if __name__ == "__main__":
    test_ttc()
