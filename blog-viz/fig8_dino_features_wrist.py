"""fig7 for the wrist camera, through the same code path.

Its own figure rather than a third row on fig7, because the two are different evidence. The
overhead view shows the backbone agreeing about a scene; the wrist view is almost entirely
gripper, so it shows the backbone agreeing about the tool, which is what the grasp depends on.

    uv run --project ../rl/calibrate python fig8_dino_features_wrist.py
"""
from __future__ import annotations

import fig7_dino_features as fig7
import sim_common as sc

if __name__ == "__main__":
    sc.save(fig7.main("arm_camera"), "fig8_dino_features_wrist.png")
