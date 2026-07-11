# SO-Frame

SO-Frame is a cheap and simple evaluation frame for SO101 arms with LeSlider add-on.

<!-- TODO: replace with a photo/render of the assembled frame -->
> 📷 _Image placeholder — add a photo/render of the assembled SO-Frame here._

## Quick links

- [Bill of Materials](#bill-of-materials)
- [CAD (Onshape)](https://cad.onshape.com/documents/e30ffd8480ec1a7673eb62f7/w/bb71304b5f8bcda7c376a401/e/bfa112e7dd1062bab3fa5a65)
- [Simulation (URDF)](urdf/README.md)

## Bill of Materials

The box frame is built entirely from **2020 (20×20 mm) T-slot aluminium extrusion**. The
[LeSlider][leslider] add-on provides the 1-DOF linear motion and has its own BOM.

### Aluminium extrusion

| Part | Length | Qty |
|------|--------|-----|
| 2020 T-slot extrusion | 1000 mm (100 cm) | 5 |
| 2020 T-slot extrusion | 500 mm (50 cm)  | 9 |

**Total: 9.5 m** of 2020 extrusion (5 × 1 m + 9 × 0.5 m).

### Brackets & handles

| Part | Qty |
|------|-----|
| 3-way hidden corner bracket (2020) | 8 |
| 2020 angle bracket (profile joiner) | 2 |
| Handle | 2 |

### Fasteners

Every bracket/handle screw pairs with a matching drop-in **T-nut**. Use **M5** or **M4** to
match your bracket & T-nut kit — 2020 kits are usually **M5** (some use **M4**); the counts
are the same either way.

| Used on | Screws / part | Screws | T-nuts |
|---------|---------------|--------|--------|
| 3-way corner bracket × 8 | 3 | 24 | 24 |
| Angle bracket × 2        | 2 | 4  | 4  |
| Handle × 2               | 2 | 4  | 4  |
| **Total** |  | **32** | **32** |

> **Verify against your kit.** Fasteners aren't all modelled in the URDF, so these are the
> typical counts: hidden 3-way brackets are assumed at 3 bolts + 3 T-nuts each. Some 3-way
> brackets instead use 6 bolts, or grub/set screws that thread into the extrusion end (no
> T-nut). Adjust to whatever hardware ships with your brackets.

### LeSlider add-on

The sliding carriage that carries the arm is the [LeSlider][leslider] mechanism (V-wheel
gantry + rack & pinion). It adds its own hardware — 4× V-wheel assemblies, 4× M5×25
low-profile screws, 4× M5 nylock nuts, eccentric spacers, and the pinion/rack — see the
[LeSlider][leslider] BOM.

[leslider]: https://github.com/TODO/LeSlider <!-- TODO: set the real LeSlider link -->


## Simulation

A ready-to-use combined URDF (`urdf/so101_on_frame.urdf`) mounts the SO-101 arm on the
frame's slider, including a wrist camera. See **[urdf/README.md](urdf/README.md)** for the
kinematics, joint limits, mounting details, and the interactive alignment helper.
