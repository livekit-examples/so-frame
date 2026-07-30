"""Import simulation/mjcf/so101_on_frame.xml via Isaac Sim's MJCF importer into a physics-correct
USD (rigid bodies, joints, collision, articulation).

Also sets joint drive stiffness/damping: the importer applies a DriveAPI to each joint but leaves
the gains at zero (it does not resolve MuJoCo's `<default class="X"><position kp="..."/>`
inheritance), and an undriven arm collapses under gravity. Values below start from the MJCF's own
`kp`: 400 for the rail, 20 for the arm/gripper, matching sts3215.xml.

Run on the Isaac Lab box, after `source ~/isaac/activate_isaac.sh`:
    python3 simulation/usd/import_mjcf.py

With these gains the default Isaac Sim physics timestep (1/60 s) is unstable, diverging within
~100 steps: it is ~8x coarser than the MJCF's validated 0.002 s. Use a matching timestep (e.g.
`World(physics_dt=1/500)`) or re-tune the gains for the consuming environment's timestep.
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

from pathlib import Path
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
from pxr import Usd, UsdPhysics

REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF_PATH = REPO_ROOT / "simulation" / "mjcf" / "so101_on_frame.xml"
OUT_DIR = REPO_ROOT / "simulation" / "usd" / "mjcf_import"
OUT_DIR.mkdir(exist_ok=True)

JOINT_GAINS = {
    "dof_slider": (400.0, 40.0),
    "shoulder_pan": (20.0, 2.0),
    "shoulder_lift": (20.0, 2.0),
    "elbow_flex": (20.0, 2.0),
    "wrist_flex": (20.0, 2.0),
    "wrist_roll": (20.0, 2.0),
    "gripper": (20.0, 2.0),
}

config = MJCFImporterConfig(
    mjcf_path=str(MJCF_PATH),
    usd_path=str(OUT_DIR),
    import_scene=False,          # robot only; this repo's own scenes add the ground/cube/bin
    merge_mesh=False,             # keep per-link mesh structure (for material transfer later)
    collision_from_visuals=False, # use our authored collision geoms (box panels), not the thin visual meshes
    collision_type="Convex Hull",
    allow_self_collision=False,
    fix_base=True,                # fixed-base rig, matching rl/environments/mjlab and rl/environments/maniskill
    run_asset_transformer=True,
    run_multi_physics_conversion=True,
)

importer = MJCFImporter(config)
usd_path = importer.import_mjcf()
print(f"Imported USD: {usd_path}")

stage = Usd.Stage.Open(usd_path)
updated = []
for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.Joint):
        continue
    name = prim.GetName()
    if name not in JOINT_GAINS:
        continue
    stiffness, damping = JOINT_GAINS[name]
    token = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
    drive = UsdPhysics.DriveAPI.Get(prim, token) or UsdPhysics.DriveAPI.Apply(prim, token)
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    updated.append((prim.GetPath(), token, stiffness, damping))

for path, token, stiffness, damping in updated:
    print(f"drive set: {path} [{token}]: stiffness={stiffness}, damping={damping}")
assert len(updated) == len(JOINT_GAINS), (
    f"expected to set drives on {len(JOINT_GAINS)} joints, only found {len(updated)}"
)

stage.GetRootLayer().Save()
print(f"Saved joint drive gains to {usd_path}")

simulation_app.close()
