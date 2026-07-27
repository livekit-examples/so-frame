"""Guards on the training/deploy contract.

These replace the old "MUST stay architecturally byte-identical to the training definitions"
comment in sim2real/policy/nets.py. That comment was the only thing keeping two hand-copied
sets of network definitions in sync, and it did not hold: the actor's proprio width drifted
from 14 to 22 in training while deploy stayed at 14.

The golden signatures below are the state_dict key/shape sets of checkpoints already trained
(verified against the pre-refactor definitions in train_squint.py and train_dino_v4.py at port
time). Changing an architecture will fail these -- which is the point: existing checkpoints
would stop loading, so it should be a deliberate edit with a format bump, not a silent one.

    uv run --project nets pytest nets/tests -q
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from soframe_nets import Actor, DinoPatchEncoder, ProprioSpec, SquintEncoder, checkpoint
from soframe_nets.proprio import ACTION_DELAY, PENDING_ACTIONS, QPOS, TARGET_QPOS

NUM_CAMS, N_ACT = 2, 7

# The v52+ layout, measured from SOFramePickPlaceBin-v1's _get_obs_agent().
LAYOUT_V52 = ProprioSpec(((QPOS, 7), (PENDING_ACTIONS, 7), (ACTION_DELAY, 1), (TARGET_QPOS, 7)))
# The pre-v52 layout every checkpoint in sim2real/policy/checkpoints/ was trained with.
LAYOUT_V51 = ProprioSpec(((QPOS, 7), (TARGET_QPOS, 7)))

SQUINT_ENCODER_KEYS = {
    "conv.0.weight": (32, 6, 4, 4), "conv.0.bias": (32,),
    "conv.2.weight": (64, 32, 4, 4), "conv.2.bias": (64,),
    "conv.4.weight": (64, 64, 3, 3), "conv.4.bias": (64,),
}


def sig(module):
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


def make(kind, res, layout):
    return checkpoint.build(kind, num_cams=NUM_CAMS, res=res, n_state=layout.n_state,
                            n_act=N_ACT, action_low=-np.ones(N_ACT), action_high=np.ones(N_ACT))


# --------------------------------------------------------------------------- architectures

def test_squint_encoder_signature_is_frozen():
    assert sig(SquintEncoder(NUM_CAMS, res=32)) == SQUINT_ENCODER_KEYS


@pytest.mark.parametrize("res,expected_tok", [(112, 128), (168, 288), (224, 512)])
def test_dino_token_count_follows_resolution(res, expected_tok):
    """n_tok = num_cams * (res/14)^2. Deploy derives it from the checkpoint's res, so this is
    what keeps a 112px checkpoint from being fed 224px tokens."""
    enc = DinoPatchEncoder(NUM_CAMS, res=res)
    assert enc.n_tok == expected_tok
    assert enc.pos_embed.shape == (1, expected_tok, 384)


def test_dino_encoder_signature_is_stable_and_excludes_the_backbone():
    enc = DinoPatchEncoder(NUM_CAMS, res=112)
    keys = sig(enc)
    assert keys["readout"] == (1, 1, 384)
    assert keys["cam_embed"] == (NUM_CAMS, 384)
    assert keys["pos_embed"] == (1, 128, 384)
    assert keys["out.0.weight"] == (512, 384)
    # 4 transformer layers, and crucially no frozen ViT weights in the checkpoint.
    assert sum(1 for k in keys if k.startswith("transformer.layers.")) == 48
    assert not [k for k in keys if "blocks" in k or "patch_embed" in k]


@pytest.mark.parametrize("kind,res,rgb_dim", [("squint", 32, 50), ("dino_patch", 112, 256)])
def test_actor_projection_width_follows_the_encoder(kind, res, rgb_dim):
    """The two encoders need different RGB widths; that used to be two copy-pasted Projection
    classes in two train scripts. build() picks it from the encoder so they cannot mismatch."""
    _, actor = make(kind, res, LAYOUT_V52)
    assert actor.proj.rgb_proj[0].out_features == rgb_dim
    assert actor.proj.repr_dim == rgb_dim + 256


def test_actor_signature_is_identical_across_encoders_except_the_projection():
    """The whole actor was duplicated per train script; only the RGB projection ever differed,
    plus the first trunk layer whose fan-in follows proj.repr_dim (50+256 vs 256+256)."""
    _, squint_actor = make("squint", 32, LAYOUT_V52)
    _, dino_actor = make("dino_patch", 112, LAYOUT_V52)
    a, b = sig(squint_actor), sig(dino_actor)
    assert set(a) == set(b)
    differing = {k for k in a if a[k] != b[k]}
    assert differing == {"proj.rgb_proj.0.weight", "proj.rgb_proj.0.bias",
                         "proj.rgb_proj.1.weight", "proj.rgb_proj.1.bias",
                         "fc.0.weight"}
    assert a["fc.0.weight"] == (256, 306) and b["fc.0.weight"] == (256, 512)


def test_actor_action_bounds_come_from_arrays_not_a_faked_env():
    actor = Actor(64, 22, 3, [-1.0, 0.0, -2.0], [1.0, 4.0, 2.0])
    assert torch.allclose(actor.action_scale, torch.tensor([1.0, 2.0, 2.0]))
    assert torch.allclose(actor.action_bias, torch.tensor([0.0, 2.0, 0.0]))


def test_unknown_encoder_kind_is_rejected():
    with pytest.raises(ValueError, match="unknown encoder kind"):
        make("dino_lora", 112, LAYOUT_V52)


# ------------------------------------------------------------------------------- preprocess

def test_squint_preprocess_is_an_area_resize_and_a_noop_at_native_res():
    enc = SquintEncoder(NUM_CAMS, res=32)
    big = torch.randint(0, 256, (2, 128, 128, 6), dtype=torch.uint8)
    want = (torch.nn.functional.interpolate(big.permute(0, 3, 1, 2).float(), size=(32, 32),
                                            mode="area").to(torch.uint8).permute(0, 2, 3, 1))
    assert torch.equal(enc.preprocess(big), want)
    small = torch.randint(0, 256, (2, 32, 32, 6), dtype=torch.uint8)
    assert enc.preprocess(small) is small


def test_squint_forward_shape():
    enc = SquintEncoder(NUM_CAMS, res=32)
    out = enc(torch.randint(0, 256, (3, 32, 32, 6), dtype=torch.uint8))
    assert out.shape == (3, enc.repr_dim) == (3, 1024)


def test_dino_head_consumes_bf16_tokens_and_returns_fp32_features():
    enc = DinoPatchEncoder(NUM_CAMS, res=112)
    out = enc(torch.randn(3, enc.n_tok, 384, dtype=torch.bfloat16))
    assert out.shape == (3, enc.repr_dim) == (3, 512)
    assert out.dtype == torch.float32


def test_dino_head_is_deterministic_in_eval_mode():
    """Deploy must get the same action for the same frame. nn.TransformerEncoder uses a fused
    kernel in eval mode that drifts ~3e-6 from the training-mode path; within eval it is exact,
    which is what a robot control loop needs."""
    enc = DinoPatchEncoder(NUM_CAMS, res=112).eval()
    x = torch.randn(2, enc.n_tok, 384, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(enc(x), enc(x))


# ---------------------------------------------------------------------------------- proprio

def test_layout_widths():
    assert LAYOUT_V52.n_state == 22
    assert LAYOUT_V51.n_state == 14
    assert LAYOUT_V52.slice_of(QPOS) == slice(0, 7)
    assert LAYOUT_V52.slice_of(PENDING_ACTIONS) == slice(7, 14)
    assert LAYOUT_V52.slice_of(ACTION_DELAY) == slice(14, 15)
    assert LAYOUT_V52.slice_of(TARGET_QPOS) == slice(15, 22)


def test_assemble_orders_fields_as_trained():
    v = LAYOUT_V52.assemble({QPOS: np.full(7, 1.0), PENDING_ACTIONS: np.full(7, 2.0),
                             ACTION_DELAY: 0.5, TARGET_QPOS: np.full(7, 3.0)})
    assert v.shape == (22,) and v.dtype == np.float32
    assert v[LAYOUT_V52.slice_of(TARGET_QPOS)].tolist() == [3.0] * 7


def test_assemble_catches_the_bug_this_module_exists_for():
    """Deploy's old build_state14 emitted qpos+target only. Against the v52 layout that is a
    loud error, not a vector with target_qpos sitting in the pending_actions slot."""
    with pytest.raises(KeyError, match="missing"):
        LAYOUT_V52.assemble({QPOS: np.zeros(7), TARGET_QPOS: np.zeros(7)})


def test_assemble_rejects_unknown_fields_and_bad_widths():
    full = {QPOS: np.zeros(7), PENDING_ACTIONS: np.zeros(7), ACTION_DELAY: 0.0,
            TARGET_QPOS: np.zeros(7)}
    with pytest.raises(KeyError, match="unexpected"):
        LAYOUT_V52.assemble({**full, "qvel": np.zeros(7)})
    with pytest.raises(ValueError, match="width 6"):
        LAYOUT_V52.assemble({**full, QPOS: np.zeros(6)})


def test_layout_measured_from_a_nested_obs_dict_matches_flatten_order():
    """from_obs_agent must reproduce mani_skill's flatten_state_dict order: insertion order,
    recursing into sub-dicts (the controller state is nested)."""
    obs = {
        "noisy_qpos": torch.zeros(4, 7),
        "pending_actions": torch.zeros(4, 7),
        "action_delay": torch.zeros(4, 1),
        "controller": {"target_qpos": torch.zeros(4, 7)},
    }
    assert ProprioSpec.from_obs_agent(obs) == LAYOUT_V52


# ------------------------------------------------------------------------------- checkpoint

@pytest.mark.parametrize("kind,res", [("squint", 32), ("dino_patch", 112)])
def test_checkpoint_roundtrip_rebuilds_from_metadata_alone(tmp_path, kind, res):
    enc, actor = make(kind, res, LAYOUT_V52)
    p = checkpoint.save(tmp_path / "ckpt.pt", enc, actor, num_cams=NUM_CAMS, res=res,
                        n_act=N_ACT, action_low=-np.ones(N_ACT), action_high=np.ones(N_ACT),
                        proprio=LAYOUT_V52, global_step=99)
    enc2, actor2, meta = checkpoint.load(p, device="cpu")
    assert meta["kind"] == kind and meta["res"] == res and meta["global_step"] == 99
    assert meta["proprio"] == LAYOUT_V52 and meta["n_state"] == 22
    assert sig(enc2) == sig(enc) and sig(actor2) == sig(actor)

    # eval() on both sides: nn.TransformerEncoder switches to a fused kernel in eval mode, so
    # the dino head's train- and eval-mode outputs differ by ~3e-6. Deploy always runs eval,
    # and load() returns eval-mode modules, so eval-vs-eval is the comparison that matters.
    enc.eval(), actor.eval()
    state = torch.randn(2, 22)
    x = (torch.randint(0, 256, (2, res, res, 6), dtype=torch.uint8) if kind == "squint"
         else torch.randn(2, enc.n_tok, 384, dtype=torch.bfloat16))
    with torch.no_grad():
        assert torch.equal(actor.get_eval_action(enc(x), state),
                           actor2.get_eval_action(enc2(x), state))


def test_save_requires_a_real_proprio_spec(tmp_path):
    enc, actor = make("squint", 32, LAYOUT_V52)
    with pytest.raises(TypeError, match="ProprioSpec"):
        checkpoint.save(tmp_path / "c.pt", enc, actor, num_cams=NUM_CAMS, res=32, n_act=N_ACT,
                        action_low=-np.ones(N_ACT), action_high=np.ones(N_ACT), proprio=22)


def test_legacy_checkpoint_refuses_to_guess_then_loads_when_told(tmp_path):
    enc, actor = make("squint", 32, LAYOUT_V51)
    p = tmp_path / "legacy.pt"
    torch.save({"encoder": enc.state_dict(), "actor": actor.state_dict(), "global_step": 7}, p)
    with pytest.raises(ValueError, match="predates the deploy checkpoint format"):
        checkpoint.load(p, device="cpu")
    enc2, actor2, meta = checkpoint.load(p, device="cpu", kind="squint", res=32,
                                         proprio=LAYOUT_V51)
    assert meta["n_state"] == 14 and meta["proprio"] == LAYOUT_V51
    assert sig(actor2) == sig(actor)
