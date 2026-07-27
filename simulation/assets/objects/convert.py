"""Convert the task objects' 3MF sources to OBJ meshes SAPIEN can load.

The 3MF files are the CAD exports (Onshape) and are the source of truth for dimensions; SAPIEN
does not read 3MF, so this writes the OBJ that the env actually loads. Re-run after replacing a
3MF and commit both.

    uv run --project ../../../rl/maniskill python convert.py

It also prints the measurements the env needs (outer extents, interior opening, wall and floor
thickness). Those are mirrored as constants in rl/maniskill/src/soframe_rl_maniskill/config.py,
and this script asserts they still match, so editing a 3MF without updating the config fails
here rather than silently changing the task.
"""

import pathlib

import numpy as np
import trimesh

HERE = pathlib.Path(__file__).resolve().parent

# What the env expects, in metres. Keep in step with config.py's CUBE_/BIN_ constants.
EXPECTED = {
    "cube": dict(extents=(0.020, 0.020, 0.020)),
    "bin": dict(extents=(0.100, 0.100, 0.030), interior_half=0.048, floor_thickness=0.002),
}


def convert(name: str) -> trimesh.Trimesh:
    mesh = trimesh.load(HERE / f"{name}.3mf", force="mesh")
    # 3MF has no material/normals worth keeping here; OBJ with recomputed normals is enough for
    # a visual mesh, and collision is built from boxes (see pick_place._load_scene).
    mesh.export(HERE / f"{name}.obj")
    return mesh


def measure(name: str, mesh: trimesh.Trimesh) -> dict:
    V = mesh.vertices
    out = dict(extents=tuple(np.round(mesh.extents, 6)),
               z_levels=tuple(np.round(np.unique(np.round(V[:, 2], 5)), 5)))
    if name == "bin":
        # Measure the interior at the floor level, not the rim. The rim carries both the inner
        # and outer profiles plus their corner fillets, so a per-vertex extent there picks up a
        # fillet rather than the flat wall face. The z == floor_thickness level contains only
        # the interior floor boundary, whose extent IS the interior half-extent.
        out["floor_thickness"] = float(out["z_levels"][1])
        floor_ring = V[np.isclose(V[:, 2], out["floor_thickness"], atol=1e-6)]
        out["interior_half"] = float(np.abs(floor_ring[:, :2]).max())
        out["outer_half"] = float(np.abs(V[:, :2]).max())
        out["interior_depth"] = float(out["z_levels"][-1] - out["floor_thickness"])
        out["wall_thickness"] = out["outer_half"] - out["interior_half"]
    return out


if __name__ == "__main__":
    for name in ("cube", "bin"):
        mesh = convert(name)
        m = measure(name, mesh)
        print(f"{name}.3mf -> {name}.obj")
        print(f"  verts {len(mesh.vertices)}  faces {len(mesh.faces)}  "
              f"watertight={mesh.is_watertight}  volume {mesh.volume * 1e6:.2f} cm^3")
        print(f"  extents (mm) {np.round(np.array(m['extents']) * 1000, 2)}")
        print(f"  z levels (mm) {np.round(np.array(m['z_levels']) * 1000, 2)}")
        if name == "bin":
            print(f"  footprint half {m['outer_half'] * 1000:.1f} mm, "
                  f"interior half {m['interior_half'] * 1000:.1f} mm "
                  f"({2 * m['interior_half'] * 1000:.0f} mm opening)")
            print(f"  wall {m['wall_thickness'] * 1000:.1f} mm, "
                  f"floor {m['floor_thickness'] * 1000:.1f} mm, "
                  f"interior depth {m['interior_depth'] * 1000:.1f} mm")

        exp = EXPECTED[name]
        assert np.allclose(m["extents"], exp["extents"], atol=1e-5), (
            f"{name}: extents {m['extents']} != expected {exp['extents']}; update config.py")
        for key in ("interior_half", "floor_thickness"):
            if key in exp:
                assert abs(m[key] - exp[key]) < 1e-5, (
                    f"{name}: {key} {m[key]} != expected {exp[key]}; update config.py")
        # The base must sit on z=0 so the env can place it by its footprint without an offset.
        assert abs(mesh.bounds[0][2]) < 1e-9, f"{name}: base is not at z=0"
        print("  OK matches config.py")
