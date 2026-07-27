# soframe-policy

The vision encoder, the actor, and the checkpoint format — everything that goes *into* a policy
checkpoint. Training (`rl/maniskill`) and deploy (`sim2real`) both depend on this package, so
these are defined once instead of copied.

Pure `torch` + `numpy`. No `mani_skill`, `gymnasium` or `livekit`, because it installs on the
robot as well as the GPU box. The critic and training loop live in `rl/maniskill` — they never
run on the robot.

## Loading a checkpoint

A checkpoint carries everything needed to rebuild itself, so there are no flags to remember:

```python
from soframe_policy import checkpoint

encoder, actor, meta = checkpoint.load("ckpt_best.pt", device="cuda")
```

`meta` has `kind`, `res`, `num_cams`, `n_act`, `proprio`, `action_low/high`, `global_step`.

## Running it

```python
state = meta["proprio"].assemble({
    "noisy_qpos": qpos,                 # measured joint positions
    "controller.target_qpos": target,   # the target being commanded
})
action = actor.get_eval_action(encoder(encoder.preprocess(rgb)), state)
```

`rgb` is the raw camera stack, `(B, H, W, 3*num_cams)` uint8. `preprocess` does whatever that
encoder needs — squinting down for the CNN, DINOv2 tokenising for the patch head — and is the
same code training uses, so sim and real cannot drift.

`assemble` raises if a field is missing, unknown, or the wrong width.

## Encoders

| `kind` | head | input |
|---|---|---|
| `squint` | CNN over a squinted image stack | `(B, res, res, 3*num_cams)` uint8, `res` ∈ {16,32,64} |
| `dino_patch` | self-attention over frozen DINOv2 patch tokens | `(B, n_tok, 384)` bf16, `n_tok = num_cams*(res/14)²` |

The frozen ViT is not in the checkpoint; it comes from `torch.hub` on first use.

## Tests

```bash
uv run --project policy pytest policy/tests -q
```

They pin each architecture's state_dict signature, so an edit that would break existing
checkpoints fails here rather than silently.
