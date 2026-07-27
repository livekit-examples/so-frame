# soframe-nets

Everything that goes *into* a policy checkpoint, defined once: the vision encoders, the SAC
actor, the proprioception layout, and the checkpoint format itself.

Both trees take this as a path dependency:

- `rl/maniskill` (training) — builds the encoder + actor, and writes checkpoints.
- `sim2real` (deploy) — rebuilds them from a checkpoint and runs them on the robot.

Pure `torch` + `numpy`. Nothing here imports `mani_skill`, `gymnasium` or `livekit`, because
this package installs on the robot's control machine as well as the GPU box. The critic,
replay buffer and training loop stay in `rl/maniskill` — they never run on the robot.

## Why it exists

Deploy used to carry hand-copied network definitions, headed by this comment:

> MUST stay architecturally byte-identical to the training definitions or the checkpoint's
> state_dict won't load.

That comment was the only mechanism keeping the two in sync, and it did not hold:

1. **The proprio layout drifted.** When per-episode action delay was added to the env, the
   actor's state vector went from `noisy_qpos(7) | target_qpos(7)` = 14 to
   `noisy_qpos(7) | pending_actions(7) | action_delay(1) | target_qpos(7)` = 22. Deploy kept
   building 14. Every checkpoint from v52 on was undeployable by that code, and if the widths
   had happened to line up it would have fed `target_qpos` into the slot the policy reads as
   `pending_actions` and driven the robot on a silently wrong observation. The action-delay
   machinery has since been removed from the env, so the contract is 14 by construction — but
   it is still recorded per checkpoint, so the next change to `_get_obs_agent` fails loudly
   rather than silently.
2. **Only 2 of 5 architectures were ever vendored**, so `dino_lora` and `dino_patch`
   checkpoints could not be deployed at all.
3. **The input resolution was a CLI flag** (`--dino-res`), remembered by hand. `v57_dino_patch`
   trained at 168; passing the 112 default loads without error and runs the policy on features
   from the wrong patch grid.

## The contract

Everything a checkpoint needs to be rebuilt is written into the checkpoint:

```python
from soframe_nets import checkpoint

encoder, actor, meta = checkpoint.load("ckpt_best.pt", device="cuda")
# meta: kind, res, num_cams, n_state, n_act, proprio (ProprioSpec), action_low/high, global_step
```

No `--arch`, no `--dino-res`, no `N_STATE = 14`. Checkpoints written before this format have no
`meta` block; `load` refuses to guess and asks for `kind`, `res` and `proprio` explicitly.

`ProprioSpec.assemble` builds the state vector in the trained field order and raises on a
missing field, an unknown field, or a wrong width:

```python
state = meta["proprio"].assemble({
    "noisy_qpos": qpos,                 # measured joint positions, sim units
    "controller.target_qpos": target,   # the integrated target we are commanding
})
```

Each encoder also owns `preprocess`, the one definition of how a raw camera stack becomes
encoder input (squint's area downsample; the DINOv2 tokenizer's resize + ImageNet
normalization + camera-major token ordering). Training calls it from an obs wrapper, deploy
calls it per tick, so they cannot disagree.

## Encoders

| `KIND` | head | input | RGB proj |
|---|---|---|---|
| `squint` | CNN over a squinted image stack | `(B, res, res, 3*num_cams)` uint8, `res` ∈ {16,32,64} | 50 |
| `dino_patch` | self-attention over frozen DINOv2 ViT-S/14 patch tokens | `(B, n_tok, 384)` bf16, `n_tok = num_cams*(res/14)²` | 256 |

The frozen ViT is **not** in the checkpoint — it is cached per device and pulled from
`torch.hub` on first use.

## Tests

```bash
uv run --project nets pytest nets/tests -q
```

The suite pins state_dict key/shape signatures for both architectures, so an architecture edit
fails loudly rather than silently invalidating existing checkpoints.
