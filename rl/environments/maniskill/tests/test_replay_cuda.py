"""The host-RAM + prefetch path, checked on a real GPU.

What host storage adds over tests/test_replay.py's synchronous CPU path: pinned staging buffers, a
background gather holding the lock against concurrent writes, an async host-to-device copy on a side
stream, and a ring of staging slots reused once their event drains. Those fail by handing back
CORRUPT batches rather than by raising, so this checks content, not liveness.

Same trick as the CPU test: every observation carries a value that uniquely identifies
(iteration, env), so a sample that mixes slots, reads a half-written row or reuses a staging buffer
before its copy landed cannot be reconciled with the reference.
"""

import pytest
import torch

from soframe_rl_maniskill.sac.replay import ObsOnlyReplay

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

NUM_ENVS = 64
ITERS = 12
CAP = NUM_ENVS * ITERS
HORIZON = 7
OBS_W = 256          # wide enough that a torn row is visible, small enough to stay quick


def _obs(k, e, width=OBS_W):
    """Observation row for (iteration k, env e): the value k*1000+e repeated across the row."""
    return torch.full((width,), 1000.0 * k + e)


def _iteration(k):
    obs = torch.stack([_obs(k, e) for e in range(NUM_ENVS)])
    state = torch.stack([_obs(k, e, 4) for e in range(NUM_ENVS)])
    acts = torch.stack([torch.tensor([float(k), float(e)]) for e in range(NUM_ENVS)])
    rews = torch.tensor([1000.0 * k + e for e in range(NUM_ENVS)]) * 0.5
    dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
    boundary = torch.full((NUM_ENVS,), (k + 1) % HORIZON == 0, dtype=torch.bool)
    return obs, state, acts, rews, dones, boundary


def _check(batch, tag=""):
    """Every row must be a real transition: successor one iteration later, same env, no seam."""
    o = batch["observations"]["rgb"]
    n = batch["next_observations"]["rgb"]
    # A row is uniform by construction; a torn read would break that before anything else.
    assert (o == o[:, :1]).all(), f"{tag}: torn observation row (staging reused too early?)"
    assert (n == n[:, :1]).all(), f"{tag}: torn successor row"

    ov, nv = o[:, 0], n[:, 0]
    k, e = torch.div(ov, 1000, rounding_mode="floor"), ov % 1000
    nk, ne = torch.div(nv, 1000, rounding_mode="floor"), nv % 1000
    assert (ne == e).all(), f"{tag}: successor belongs to a different env"
    assert (nk == k + 1).all(), f"{tag}: successor is not one iteration later"
    assert (((k + 1) % HORIZON) != 0).all(), f"{tag}: a boundary transition was sampled"

    st = batch["observations"]["state"][:, 0]
    assert (st == ov).all(), f"{tag}: state does not match the rgb of the same slot"
    assert (batch["actions"][:, 0] == k).all(), f"{tag}: action k does not match its slot"
    assert (batch["actions"][:, 1] == e).all(), f"{tag}: action e does not match its slot"
    assert torch.allclose(batch["rewards"], ov * 0.5), f"{tag}: reward does not match its slot"


def _fill(rb, n_iters, device):
    for k in range(n_iters):
        obs, state, acts, rews, dones, boundary = _iteration(k)
        rb.extend(obs.to(device), state.to(device), acts.to(device),
                  rews.to(device), dones.to(device), boundary.to(device))


def test_host_storage_keeps_the_buffer_off_the_gpu():
    rb = ObsOnlyReplay(CAP, NUM_ENVS, device="cuda", storage_device="cpu", prefetch=2)
    try:
        _fill(rb, 3, "cuda")
        assert rb._rgb.device.type == "cpu", "storage should be on the host"
        assert rb._rgb.is_pinned(), "host storage must be pinned or the copy is not async"
        batch = rb.sample(128)
        assert batch["observations"]["rgb"].device.type == "cuda", "batches must land on the GPU"
        _check(batch, "host->gpu")
    finally:
        rb.close()


def test_prefetched_batches_are_correct_under_concurrent_writes():
    """The real test: keep extending while the prefetch thread gathers behind us.

    Writes recycle slots continuously, so a batch whose gather raced a write, or whose staging
    slot was refilled before its copy landed, shows up as a value that fails _check.
    """
    rb = ObsOnlyReplay(CAP, NUM_ENVS, device="cuda", storage_device="cpu", prefetch=3)
    try:
        _fill(rb, 3, "cuda")
        k = 3
        for _ in range(80):                       # ~7 full wraps of a 12-iteration buffer
            batch = rb.sample(256)
            _check(batch, f"iter {k}")
            obs, state, acts, rews, dones, boundary = _iteration(k)
            rb.extend(obs.cuda(), state.cuda(), acts.cuda(), rews.cuda(),
                      dones.cuda(), boundary.cuda())
            k += 1
        torch.cuda.synchronize()
    finally:
        rb.close()


def test_prefetch_off_matches_prefetch_on():
    """prefetch=0 takes the synchronous path; both must satisfy the same invariant."""
    for prefetch in (0, 2):
        rb = ObsOnlyReplay(CAP, NUM_ENVS, device="cuda", storage_device="cpu", prefetch=prefetch)
        try:
            _fill(rb, 9, "cuda")
            for _ in range(10):
                _check(rb.sample(256), f"prefetch={prefetch}")
        finally:
            rb.close()


def test_batch_size_change_is_rejected_not_silently_wrong():
    # Staging is allocated once, so a changed batch size must fail loudly.
    rb = ObsOnlyReplay(CAP, NUM_ENVS, device="cuda", storage_device="cpu", prefetch=2)
    try:
        _fill(rb, 3, "cuda")
        rb.sample(128)
        with pytest.raises(ValueError, match="bound to batch_size"):
            rb.sample(256)
    finally:
        rb.close()


def test_close_is_idempotent_and_safe_before_first_sample():
    rb = ObsOnlyReplay(CAP, NUM_ENVS, device="cuda", storage_device="cpu", prefetch=2)
    rb.close()
    rb.close()
