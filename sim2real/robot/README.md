# robot

Robot-side runtime for the SO-101-on-frame rig. Connects the 6-motor SO-101
arm, its two cameras (arm + overhead), and the linear rail (`dof_slider`); joins
the LiveKit room as `robot`; publishes 7-DOF state + RAW video every tick; and
applies the latest incoming Portal action (absolute position targets).

```bash
# from sim2real/ (single uv project)
cp .env.example .env    # fill LIVEKIT_* and SO101_*
uv sync
uv run robot
```

The wire schema is `../portal.yaml` (7 joints, 2 cameras), shared with the
policy. State/actions are in REAL units (arm/gripper degrees, rail mm); the
policy owns the sim<->real bridge.

## The rail

lerobot's `SO101Follower` models only the 6 arm motors. The 7th DOF (the rail)
plugs in through `slider.py` (`SliderActuator`). It ships with a `StubSlider`
no-op so the whole pipeline can be brought up before the rail is wired; swap in
the real driver there. See `slider.py` (`NEEDS-HARDWARE`).

## Raw frames, not the sim view

Unlike a naive deploy, the robot publishes the cameras' RAW wide-FOV frames. The
policy operator rectifies them into the narrow sim view itself (via the saved
camera mapping), so the robot stays policy-agnostic and a web-ui still renders
the true stream. See `../policy/README.md`.
