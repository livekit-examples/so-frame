# Running this on AMD (RDNA3)

This `amd` branch runs SO-101-on-frame vision-based RL on an AMD Radeon GPU, using the
[ManiSkill AMD port](../../../../ManiSkill/tools/amd_port/REPRODUCE.md) — PhysX 5's GPU solver
recompiled from CUDA to HIP, plus SAPIEN's Vulkan↔compute interop ported to HIP so `obs_mode="rgb"`
works.

**Nothing in this project's own code changes.** The only difference from `main` is where `torch`
comes from: `pyproject.toml` points it at PyTorch's ROCm index. `mani_skill` stays the unmodified
3.0.1 release from PyPI, because the port lives in PhysX and SAPIEN rather than in ManiSkill — so
there is no fork to track.

## Setup

```bash
# 1. Build the ManiSkill AMD port once (~20 min with the prebuilt gfx1100 kernels)
cd /path/to/ManiSkill/benchmarks && uv sync --project amd && cd ..
./tools/amd_port/bootstrap.sh --prebuilt --fix-torch-rocm --venv benchmarks/amd/.venv

# 2. This project's environment (ROCm torch comes from the index configured in pyproject.toml)
cd /path/to/so-frame/rl/environments/maniskill
uv sync

# 3. Drop the AMD physics + renderer underneath it
/path/to/ManiSkill/tools/amd_port/install-into.sh .venv --fix-torch-rocm
source ~/maniskill-amd/env.sh

# 4. Train — obs_mode="rgb" and sim_backend="gpu" are the defaults and both work
uv run python train.py
```

## Things that will bite you

- **Re-run `install-into.sh` after any `uv sync`.** It overwrites the `sapien` package's shared
  libraries, and `uv sync` restores the stock CUDA-linked ones. Symptom:
  `failed to find device "cuda"`.
- **torch must come from the ROCm index.** The default PyPI `torch` is a CUDA build and will not see
  the GPU. `pyproject.toml` handles this; do not "simplify" the `[tool.uv.sources]` block away.
- **torch is pinned `<2.8`** deliberately — see the comment in `pyproject.toml`. The rocm6.2 index
  tops out at 2.5.1, which does not satisfy `tensordict`/`torchrl` 0.7.2, hence rocm6.3.
- **Rendering works, but its synchronisation is host-side on AMD**, because ROCm cannot import Vulkan
  timeline semaphores. Render and compute therefore do not overlap, so visual training is slower than
  the ~1.67× physics gap alone would predict. Measure it rather than extrapolating.
- **Requires RDNA3 (gfx1100).** CDNA (MI200/MI300) is wave64 and unsupported — the port's build
  refuses it rather than produce silently wrong physics.

## What is verified

Physics on AMD is validated in detail (15/15 checks, bit-identical determinism, GPU-vs-CPU
divergence matching NVIDIA to three significant figures, and a policy trained on AMD scoring 98.4%
on NVIDIA). `obs_mode="rgb"` is verified to return real frames.

**This branch's own training run has not been executed on AMD yet** — the dependency set resolves and
the underlying pieces are tested, but nobody has watched `train.py` learn on a Radeon. Treat the
first run as such.
