"""Circular replay that stores each observation once instead of twice, optionally in host RAM.

The next observation of a transition is just the observation the same env reports one step later, so
storing ``next_observations`` alongside ``observations`` doubles memory for no information. That
matters for the DINOv2 patch head: a 2-camera 168px token grid is 216 KB, so a two-copy layout costs
432 KB per transition.

``extend`` is called once per iteration with exactly ``num_envs`` transitions and storage is
round-robin, so the slot layout is predictable:

    slot(iteration k, env e) = (k mod iters) * num_envs + e

The successor of slot ``i`` is therefore always ``(i + num_envs) % capacity``: same env, one
iteration later. Two slot classes cannot be read that way and are masked out of sampling:

* the newest iteration, whose successor has not been written yet;
* truncation/termination boundaries, where the true next observation is the pre-reset
  ``final_observation`` and the slot one stride later holds the POST-reset observation of the next
  episode. Bootstrapping across that seam is the classic replay bug, so those transitions are
  dropped rather than approximated (~1 in EPISODE_HORIZON with ``partial_reset`` off).

Wraparound needs no special handling: writing iteration k overwrites iteration ``k - iters``, whose
own predecessor was already overwritten one iteration earlier, so no surviving slot is left pointing
at a recycled successor.

Sampling is uniform with replacement over the valid slots, matching torchrl's ``RandomSampler``.

Host storage
------------
``storage_device="cpu"`` keeps the buffer in pinned host RAM instead of VRAM, trading capacity for
bandwidth: every update pulls ``batch * (obs + next_obs)`` over PCIe, and the CPU-side gather of
scattered rows is single-threaded memcpy. A prefetch thread hides both: it holds the lock only for
the gather, stages into pinned buffers, issues the host-to-device copy on a side stream, and hands
the consumer a CUDA event to wait on.

Prefetch engages only for the pinned-host-to-CUDA case. With storage and compute on the same device
there is nothing to overlap, so ``sample`` takes the direct, synchronous path, which also keeps the
unit tests deterministic.
"""

from __future__ import annotations

import queue
import threading

import torch
from tensordict import TensorDict


class ObsOnlyReplay:
    """Round-robin replay buffer storing one copy of each observation.

    Args:
        capacity: requested number of transitions. Rounded DOWN to a whole multiple of
            ``num_envs`` so the successor stride is exact.
        num_envs: transitions per ``extend`` call. The successor stride.
        device: device sampled batches are delivered on (where training runs).
        storage_device: where the buffer itself lives. Defaults to ``device``; pass ``"cpu"`` for
            pinned host RAM.
        prefetch: batches kept in flight by the background thread. Only used when storage is
            pinned host memory and ``device`` is CUDA. 0 disables the thread.
    """

    def __init__(self, capacity: int, num_envs: int, device, storage_device=None,
                 prefetch: int = 2):
        if capacity < 2 * num_envs:
            raise ValueError(
                f"capacity {capacity} must hold at least two iterations of {num_envs} envs; "
                "with fewer, no slot ever has a successor and nothing is sampleable"
            )
        self.num_envs = int(num_envs)
        self.iters = int(capacity) // self.num_envs
        self.capacity = self.iters * self.num_envs
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device if storage_device is not None else device)

        # Pinned host pages are what make the copy asynchronous: unpinned, torch bounces through its
        # own staging buffer and the transfer serialises against compute.
        self._pinned = self.storage_device.type == "cpu" and self.device.type == "cuda"
        self._async = self._pinned and prefetch > 0
        self._prefetch = int(prefetch) if self._async else 0

        def _z(*shape, dtype):
            return torch.zeros(*shape, device=self.storage_device, dtype=dtype,
                               pin_memory=self._pinned)

        self._rgb = None      # lazily allocated from the first batch's dtype/shape
        self._state = None
        self._actions = None
        self._rewards = _z(self.capacity, dtype=torch.float32)
        self._dones = _z(self.capacity, dtype=torch.bool)
        # A boundary slot's stored successor belongs to the next episode, so it is never valid.
        self._boundary = _z(self.capacity, dtype=torch.bool)
        self._valid = _z(self.capacity, dtype=torch.bool)

        self._cursor = 0          # slot where the next iteration is written
        self._written = 0         # iterations written, ever
        self._valid_idx = None    # cache, refreshed once per extend rather than per sample

        # Writers and the prefetch thread both touch the store; the lock is held for the whole
        # gather so a slot cannot be recycled underneath a batch that is mid-copy.
        self._lock = threading.Lock()
        self._q: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream = None
        self._stage: list | None = None
        self._batch_size = None

    # -- capacity accounting -----------------------------------------------------------
    def __len__(self):
        return min(self._written, self.iters) * self.num_envs

    @property
    def n_valid(self):
        return 0 if self._valid_idx is None else self._valid_idx.numel()

    def _alloc(self, rgb, state, actions):
        def _like(sample, dtype=None):
            return torch.zeros((self.capacity, *sample.shape[1:]), device=self.storage_device,
                               dtype=dtype or sample.dtype, pin_memory=self._pinned)
        self._rgb = _like(rgb)
        self._state = _like(state)
        self._actions = _like(actions)

    def extend(self, obs_rgb, obs_state, actions, rewards, dones, boundary):
        """Append one iteration of ``num_envs`` transitions.

        ``boundary`` marks envs whose stored successor observation belongs to the next episode, i.e.
        where the loop substituted ``final_observation`` into its next_obs. MUST be the same mask the
        loop uses for that substitution.
        """
        n = self.num_envs
        if obs_rgb.shape[0] != n:
            raise ValueError(f"expected {n} transitions per extend, got {obs_rgb.shape[0]}")
        if self._rgb is None:
            self._alloc(obs_rgb, obs_state, actions)

        start = self._cursor
        idx = torch.arange(start, start + n, device=self.storage_device) % self.capacity

        # Blocking device-to-host copies. One iteration is num_envs rows (110 MB for dino at 512
        # envs, ~6 ms), small enough that overlapping the write is not worth the lifetime rules it
        # would put on the caller's tensors.
        with self._lock:
            self._rgb[idx] = obs_rgb.to(device=self.storage_device, dtype=self._rgb.dtype)
            self._state[idx] = obs_state.to(device=self.storage_device, dtype=self._state.dtype)
            self._actions[idx] = actions.to(device=self.storage_device, dtype=self._actions.dtype)
            self._rewards[idx] = rewards.reshape(-1).to(self.storage_device, torch.float32)
            self._dones[idx] = dones.reshape(-1).to(self.storage_device, torch.bool)
            self._boundary[idx] = boundary.reshape(-1).to(self.storage_device, torch.bool)

            # This iteration has no successor yet; the previous one just got its successor.
            self._valid[idx] = False
            if self._written >= 1:
                prev = (idx - n) % self.capacity
                self._valid[prev] = ~self._boundary[prev]

            self._cursor = (start + n) % self.capacity
            self._written += 1
            self._valid_idx = self._valid.nonzero(as_tuple=True)[0]

    # -- sampling ----------------------------------------------------------------------
    def _pick(self, batch_size):
        """Indices of a uniform sample with replacement, and their successors."""
        pick = torch.randint(self.n_valid, (batch_size,), device=self.storage_device)
        i = self._valid_idx[pick]
        return i, (i + self.num_envs) % self.capacity

    def _pack(self, rgb_i, state_i, rgb_j, state_j, act, rew, done, batch_size, device):
        return TensorDict(
            observations=TensorDict(rgb=rgb_i, state=state_i,
                                    batch_size=batch_size, device=device),
            next_observations=TensorDict(rgb=rgb_j, state=state_j,
                                         batch_size=batch_size, device=device),
            actions=act,
            rewards=rew,
            dones=done,
            batch_size=batch_size,
            device=device,
        )

    def _sample_direct(self, batch_size):
        """Storage and compute on the same device: plain fancy-indexing, no staging."""
        with self._lock:
            i, j = self._pick(batch_size)
            return self._pack(self._rgb[i], self._state[i], self._rgb[j], self._state[j],
                              self._actions[i], self._rewards[i], self._dones[i],
                              batch_size, self.device)

    def _make_stage(self, batch_size):
        """One set of pinned staging tensors, sized for a batch's obs and successor."""
        def _p(src, n):
            return torch.zeros((n, *src.shape[1:]), dtype=src.dtype, pin_memory=True)
        return dict(
            rgb=_p(self._rgb, 2 * batch_size),        # [obs | next_obs], one gather each
            state=_p(self._state, 2 * batch_size),
            actions=_p(self._actions, batch_size),
            rewards=_p(self._rewards, batch_size),
            dones=_p(self._dones, batch_size),
            event=torch.cuda.Event(),
            used=False,
        )

    def _fill_and_ship(self, stage, batch_size):
        """Gather into pinned staging, then push to the GPU on the side stream.

        Returns the device-side TensorDict plus the event that marks the copy complete. The
        caller must make its stream wait on that event before touching the tensors.
        """
        # Reusing a staging slot is only safe once its previous transfer has drained.
        if stage["used"]:
            stage["event"].synchronize()

        with self._lock:
            i, j = self._pick(batch_size)
            ij = torch.cat([i, j])
            torch.index_select(self._rgb, 0, ij, out=stage["rgb"])
            torch.index_select(self._state, 0, ij, out=stage["state"])
            torch.index_select(self._actions, 0, i, out=stage["actions"])
            torch.index_select(self._rewards, 0, i, out=stage["rewards"])
            torch.index_select(self._dones, 0, i, out=stage["dones"])

        with torch.cuda.stream(self._stream):
            g = {k: stage[k].to(self.device, non_blocking=True)
                 for k in ("rgb", "state", "actions", "rewards", "dones")}
            stage["event"].record(self._stream)
        stage["used"] = True

        td = self._pack(g["rgb"][:batch_size], g["state"][:batch_size],
                        g["rgb"][batch_size:], g["state"][batch_size:],
                        g["actions"], g["rewards"], g["dones"], batch_size, self.device)
        return td, stage["event"]

    def _worker(self, batch_size):
        s = 0
        while not self._stop.is_set():
            try:
                item = self._fill_and_ship(self._stage[s], batch_size)
            except Exception as exc:                      # surface it on the consumer's thread
                self._q.put(exc)
                return
            s = (s + 1) % len(self._stage)
            while not self._stop.is_set():
                try:
                    self._q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def _start(self, batch_size):
        self._batch_size = batch_size
        self._stream = torch.cuda.Stream(device=self.device)
        # One staging set per in-flight batch, plus one being filled.
        self._stage = [self._make_stage(batch_size) for _ in range(self._prefetch + 1)]
        self._q = queue.Queue(maxsize=self._prefetch)
        self._thread = threading.Thread(target=self._worker, args=(batch_size,), daemon=True,
                                        name="replay-prefetch")
        self._thread.start()

    def sample(self, batch_size: int) -> TensorDict:
        """Uniform sample with replacement over valid slots, shaped like the old buffer's."""
        if self.n_valid == 0:
            raise RuntimeError("no sampleable transitions yet (need at least two iterations)")
        if not self._async:
            return self._sample_direct(batch_size)

        if self._thread is None:
            self._start(batch_size)
        elif batch_size != self._batch_size:
            raise ValueError(
                f"prefetch is bound to batch_size {self._batch_size}; got {batch_size}. "
                "The staging buffers are allocated once, so the size cannot change mid-run."
            )
        item = self._q.get()
        if isinstance(item, Exception):
            raise item
        td, event = item
        torch.cuda.current_stream(self.device).wait_event(event)
        return td

    def close(self):
        """Stop the prefetch thread. Idempotent; safe to call on a buffer that never started."""
        self._stop.set()
        if self._thread is not None:
            while not self._q.empty():                    # unblock a worker parked on put()
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break
            self._thread.join(timeout=5)
            self._thread = None

    def bytes_per_transition(self) -> int:
        """Allocated bytes per slot, for the sizing arithmetic in the README/args docs."""
        if self._rgb is None:
            return 0
        per = self._rgb[0].numel() * self._rgb.element_size()
        per += self._state[0].numel() * self._state.element_size()
        per += self._actions[0].numel() * self._actions.element_size()
        per += 4 + 1 + 1 + 1          # reward, done, boundary, valid
        return per
