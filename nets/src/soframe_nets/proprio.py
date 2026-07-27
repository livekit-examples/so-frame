"""The proprioception contract: which fields the actor's state vector holds, in which order.

The actor's state vector is ``flatten_state_dict(env._get_obs_agent())``, so its layout is the
insertion order of that dict. The current contract is deliberately the smallest thing the real
rig can always produce:

    noisy_qpos(7) | controller.target_qpos(7)  = 14

This module exists because that layout silently changed once and deploy never noticed. Adding
per-episode action delay to the env inserted two fields into the *middle* of the vector
(``pending_actions(7)`` and ``action_delay(1)``, giving 22), while deploy went on hardcoding
``N_STATE = 14`` and building ``qpos(7) + target(7)``. Against a 22-dim checkpoint that fails
to load outright; had the widths happened to line up it would have fed ``target_qpos`` into the
slot the policy reads as ``pending_actions`` and driven the robot on a silently wrong
observation. The action-delay machinery has since been removed from the env, so 14 is now true
by construction rather than by a config knob.

The layout still travels inside the checkpoint: training measures it from the live env, deploy
assembles its vector through ``assemble``, and a field that is missing, extra or the wrong
width raises at startup instead of quietly misaligning. That is what makes a future change to
``_get_obs_agent`` a loud failure rather than a robot driving on garbage.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Field names the env emits, spelled once so training and deploy agree.
QPOS = "noisy_qpos"
TARGET_QPOS = "controller.target_qpos"


def _walk(state_dict, prefix=""):
    """Yield (dotted_name, width) in flatten_state_dict order (insertion order, recursive)."""
    for key, value in state_dict.items():
        if isinstance(value, dict):
            yield from _walk(value, prefix + key + ".")
            continue
        shape = tuple(getattr(value, "shape", ()))
        # Batched (num_envs, width); a scalar-per-env field is width 1.
        width = int(shape[-1]) if len(shape) > 1 else 1
        yield prefix + key, width


@dataclass(frozen=True)
class ProprioSpec:
    """Ordered (name, width) layout of the actor's proprio vector."""

    fields: tuple[tuple[str, int], ...]

    @property
    def n_state(self) -> int:
        return sum(w for _, w in self.fields)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(n for n, _ in self.fields)

    def width_of(self, name: str) -> int:
        for n, w in self.fields:
            if n == name:
                return w
        raise KeyError(f"{name!r} not in proprio layout {self.names}")

    def slice_of(self, name: str) -> slice:
        off = 0
        for n, w in self.fields:
            if n == name:
                return slice(off, off + w)
            off += w
        raise KeyError(f"{name!r} not in proprio layout {self.names}")

    @classmethod
    def from_obs_agent(cls, obs_agent: dict) -> "ProprioSpec":
        """Measure the layout from a live env's ``_get_obs_agent()`` output (training side)."""
        return cls(tuple(_walk(obs_agent)))

    def assemble(self, values, dtype=np.float32) -> np.ndarray:
        """Build the state vector from a {field_name: value} mapping, in the trained order.

        Raises on a missing field, an unknown field, or a width mismatch -- the three ways
        deploy can drift from training without a crash to point at it.
        """
        missing = [n for n in self.names if n not in values]
        extra = [n for n in values if n not in self.names]
        if missing or extra:
            raise KeyError(
                "proprio mismatch against the checkpoint's layout "
                f"{list(self.fields)}:"
                + (f" missing {missing}" if missing else "")
                + (f" unexpected {extra}" if extra else "")
            )
        parts = []
        for name, width in self.fields:
            arr = np.atleast_1d(np.asarray(values[name], dtype=dtype)).ravel()
            if arr.size != width:
                raise ValueError(
                    f"proprio field {name!r} has width {arr.size}, checkpoint expects {width}"
                )
            parts.append(arr)
        out = np.concatenate(parts) if parts else np.zeros(0, dtype=dtype)
        assert out.size == self.n_state, (out.size, self.n_state)
        return out.astype(dtype, copy=False)

    def to_meta(self) -> list[list]:
        return [[n, w] for n, w in self.fields]

    @classmethod
    def from_meta(cls, meta) -> "ProprioSpec":
        return cls(tuple((str(n), int(w)) for n, w in meta))

    def describe(self) -> str:
        out, off = [], 0
        for n, w in self.fields:
            out.append(f"[{off}:{off + w}] {n}")
            off += w
        return f"n_state={self.n_state}: " + "  ".join(out)
