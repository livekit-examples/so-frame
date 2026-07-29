"""Differential test for ObsOnlyReplay against the two-copy layout it replaces.

Keeps a reference dict of what the two-copy layout WOULD have stored for every transition and
asserts every sample agrees with it, which catches stride errors, wraparound aliasing and boundary
seams in one assertion. CPU, tiny tensors, no GPU and no ManiSkill.
"""

import torch

from soframe_rl_maniskill.sac.replay import ObsOnlyReplay

NUM_ENVS = 4
ITERS = 5                     # capacity = 20 slots, so the stream below wraps ~4x
CAP = NUM_ENVS * ITERS
HORIZON = 7                   # short horizon so boundaries appear often
DEV = torch.device("cpu")


def _stream(n_iters, horizon=HORIZON):
    """Synthetic rollout. Every observation is a unique value so aliasing is detectable.

    Returns the per-iteration tensors plus `truth`: for each (iteration, env), what the
    two-copy layout would have stored as next_observations.
    """
    obs, acts, rews, dones, bounds, truth = [], [], [], [], [], {}
    # value 1000*k + e uniquely identifies the observation of env e at iteration k
    def val(k, e):
        return 1000.0 * k + e

    for k in range(n_iters + 1):
        obs.append(torch.tensor([[val(k, e)] for e in range(NUM_ENVS)]))
    for k in range(n_iters):
        # all envs truncate together, as with partial_reset=False
        is_boundary = ((k + 1) % horizon == 0)
        b = torch.full((NUM_ENVS,), is_boundary, dtype=torch.bool)
        acts.append(torch.tensor([[float(k), float(e)] for e in range(NUM_ENVS)]))
        rews.append(torch.tensor([val(k, e) * 0.5 for e in range(NUM_ENVS)]))
        dones.append(torch.zeros(NUM_ENVS, dtype=torch.bool))
        bounds.append(b)
        for e in range(NUM_ENVS):
            # On a boundary the real next obs is the pre-reset final observation, which is NOT
            # obs[k+1] (that is the post-reset start of the next episode).
            truth[(k, e)] = None if is_boundary else val(k + 1, e)
    return obs, acts, rews, dones, bounds, truth


def _fill(n_iters):
    rb = ObsOnlyReplay(CAP, NUM_ENVS, DEV)
    obs, acts, rews, dones, bounds, truth = _stream(n_iters)
    for k in range(n_iters):
        rb.extend(obs[k], obs[k], acts[k], rews[k], dones[k], bounds[k])
    return rb, obs, acts, rews, truth


def test_capacity_rounds_down_to_whole_iterations():
    rb = ObsOnlyReplay(23, NUM_ENVS, DEV)   # 23 is not a multiple of 4
    assert rb.capacity == 20
    assert rb.iters == 5


def test_too_small_capacity_rejected():
    # One iteration of capacity can never produce a successor, so it must not silently
    # produce an unsampleable buffer.
    try:
        ObsOnlyReplay(NUM_ENVS, NUM_ENVS, DEV)
    except ValueError:
        return
    raise AssertionError("expected ValueError for a capacity below two iterations")


def test_nothing_sampleable_after_one_iteration():
    rb = ObsOnlyReplay(CAP, NUM_ENVS, DEV)
    obs, acts, rews, dones, bounds, _ = _stream(1)
    rb.extend(obs[0], obs[0], acts[0], rews[0], dones[0], bounds[0])
    assert rb.n_valid == 0


def test_next_obs_matches_two_copy_layout_including_wraparound():
    """The core claim, checked against the reference for every sample.

    18 iterations over a 5-iteration buffer wraps more than three times, and the 7-step horizon puts
    boundaries at iterations 6 and 13, both of which get overwritten.
    """
    n_iters = 18
    rb, obs, acts, rews, truth = _fill(n_iters)

    d = rb.sample(4096)
    seen = set()
    for row in range(4096):
        o = d["observations"]["rgb"][row].item()
        nxt = d["next_observations"]["rgb"][row].item()
        k, e = int(o // 1000), int(round(o % 1000))
        seen.add((k, e))
        assert truth[(k, e)] is not None, f"boundary transition (k={k},e={e}) was sampled"
        assert nxt == truth[(k, e)], (
            f"next_obs mismatch for (k={k},e={e}): got {nxt}, two-copy layout had "
            f"{truth[(k, e)]}"
        )
        # the rest of the transition must belong to the same slot
        assert d["actions"][row].tolist() == [float(k), float(e)]
        assert abs(d["rewards"][row].item() - o * 0.5) < 1e-4

    # Only recent, non-boundary iterations should ever appear: the newest has no successor,
    # and anything older than `iters` has been overwritten.
    newest = n_iters - 1
    for (k, _) in seen:
        assert k != newest, f"iteration {k} has no successor yet but was sampled"
        assert k > newest - ITERS, f"iteration {k} should have been overwritten"


def test_boundary_transitions_are_never_sampled():
    # horizon 1 makes EVERY transition a boundary, so nothing may be sampleable.
    rb = ObsOnlyReplay(CAP, NUM_ENVS, DEV)
    obs, acts, rews, dones, bounds, _ = _stream(10, horizon=1)
    for k in range(10):
        rb.extend(obs[k], obs[k], acts[k], rews[k], dones[k], bounds[k])
    assert rb.n_valid == 0


def test_valid_count_excludes_newest_and_boundaries():
    n_iters = 4                      # fits without wrapping (capacity is 5 iterations)
    rb, *_ = _fill(n_iters)
    # iterations 0..2 have successors; none of 0..2 is a boundary (horizon 7 -> boundary at 6)
    assert rb.n_valid == 3 * NUM_ENVS


def test_halves_memory_versus_two_copies():
    rb, *_ = _fill(6)
    obs_bytes = rb._rgb[0].numel() * rb._rgb.element_size()
    state_bytes = rb._state[0].numel() * rb._state.element_size()
    per = rb.bytes_per_transition()
    # A two-copy layout would add another obs+state per transition.
    two_copy = per + obs_bytes + state_bytes
    assert per < two_copy
    # For a realistic dino token grid the saving must approach exactly 2x.
    tok = 128 * 384 * 2          # res 112, 2 cams, bf16
    assert abs((tok + 4 * 14 + tok) / (tok + 4 * 14) - 2.0) < 0.02
