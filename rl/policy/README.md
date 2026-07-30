# soframe-policy

The vision encoders, the actor, and the checkpoint format: everything that goes *into* a policy
checkpoint. Training (`rl/environments/maniskill`) and deploy (`rl/deploy`) both depend on this package, so
these are defined once instead of copied.

Pure `torch` + `numpy`. No `mani_skill`, `gymnasium` or `livekit`, because it installs on the
robot as well as the GPU box. The critic and training loop live in `rl/environments/maniskill`, they never
run on the robot.

## Loading a checkpoint

A checkpoint carries everything needed to rebuild itself, so there are no flags to remember:

```python
from soframe_policy import checkpoint

encoder, actor, meta = checkpoint.load("ckpt_best.pt", device="cuda")
```

`meta` has `format_version`, `kind`, `res`, `num_cams`, `n_state`, `n_act`, `proprio`,
`action_low/high`, `encoder_kwargs`, `global_step`.

## Running it

```python
state = torch.as_tensor(meta["proprio"].assemble({
    "noisy_qpos": qpos,                 # measured joint positions
    "controller.target_qpos": target,   # the target being commanded
})).unsqueeze(0)                        # (1, 14) float32
action = actor.get_eval_action(encoder(encoder.preprocess(rgb)), state)
```

`rgb` is the raw camera stack, `(B, H, W, 3*num_cams)` uint8. `preprocess` does whatever that
encoder needs (squinting down for the CNN, DINOv2 tokenising for the patch head, DINOv2 pooling
for the global head) and is the same code training uses, so sim and real cannot drift on resize,
normalization or camera order.

The proprio contract is `noisy_qpos(7) | controller.target_qpos(7)` = 14, measured from the live
training env and recorded per checkpoint, so deploy assembles its vector in the trained order
rather than assuming a width. `assemble` raises if a field is missing, unknown, or the wrong width.

## Encoders

| `kind` | head | input |
|---|---|---|
| `squint` | CNN over a squinted image stack | `(B, res, res, 3*num_cams)` uint8, `res` ∈ {16,32,64} |
| `dino_patch` | self-attention over frozen DINOv2 patch tokens | `(B, n_tok, 384)` bf16, `n_tok = num_cams*(res/14)²` |
| `dino_global` | MLP over one frozen DINOv2 vector per camera | `(B, n_vec, 384)` bf16, `n_vec = num_cams` (×2 for `pool="cls_mean"`) |

`dino_global` is `dino_patch`'s collapsed-pooling control: same frozen ViT-S/14 with registers,
same resolution, same head width, so whether the patch grid survives is the only difference.

`Encoder.RGB_PROJ_DIM` sets the actor's RGB bottleneck: 50 for the squint CNN, 256 for both
DINOv2 heads, matched between the two so a comparison of them is not a capacity difference.

`dino_global`'s `pool` is recorded in the checkpoint, under `encoder_kwargs`: `cls` and `mean`
produce identical state-dict shapes, so it cannot be recovered from the weights.

The frozen ViT is not in the checkpoint; it comes from `torch.hub` on first use.

## Commands

Run the contract tests. They pin the squint CNN's and the patch head's state_dict signatures, the
actor's projection widths, the 14-dim proprio layout and the checkpoint roundtrip, so an edit that
would break existing checkpoints fails here rather than silently. 23 tests, about a second:

```bash
uv run --project rl/policy pytest rl/policy/tests -q
```
