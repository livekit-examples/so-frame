# Running this on AMD (RDNA3)

This `amd` branch runs SO-101-on-frame vision-based RL on an AMD Radeon GPU, using the
[ManiSkill AMD port](https://github.com/pham-tuan-binh/ManiSkill/blob/amd-port-spec/REPRODUCE.md):
PhysX 5's GPU solver recompiled from CUDA to HIP, plus SAPIEN's Vulkan↔compute interop ported to HIP
so `obs_mode="rgb"` works.

**Nothing in this project's own code changes.** The only difference from `main` is dependency pins.
`mani_skill` stays the unmodified 3.0.1 release from PyPI, because the port lives in PhysX and
SAPIEN rather than in ManiSkill, so there is no fork to track.

## Setup

```bash
# 1. Build the ManiSkill AMD port once (~20 min with the prebuilt gfx1100 kernels)
cd /path/to/ManiSkill/benchmarks && uv sync --project amd && cd ..
./tools/amd_port/bootstrap.sh --prebuilt --fix-torch-rocm --venv benchmarks/amd/.venv

# 2. This project's environment (ROCm torch comes from the index configured in pyproject.toml)
cd /path/to/so-frame/rl/environments/maniskill
uv sync

# 3. Drop the AMD physics + renderer underneath it. --fix-torch-rocm is REQUIRED, see below.
/path/to/ManiSkill/tools/amd_port/install-into.sh .venv --fix-torch-rocm

# 4. Train. Every flag here is required on AMD; see "Things that will bite you".
ulimit -n 65535
WANDB_MODE=offline uv run python train.py \
    --encoder dino_patch --replay-episodes 0.5 \
    --no-compile --no-cudagraphs \
    --track --wandb-project-name <project> --exp-name <run>

# afterwards (or periodically during the run), push the metrics to the cloud:
uv run wandb sync wandb/offline-run-<id>
```

No `LD_LIBRARY_PATH` step is needed: the port's libraries carry an `$ORIGIN` rpath.

## Things that will bite you

- **Re-run `install-into.sh` after any `uv sync`.** It overwrites the `sapien` package's shared
  libraries, and `uv sync` restores the stock CUDA-linked ones. Symptom:
  `failed to find device "cuda"`.

- **`--fix-torch-rocm` is required, even though the ROCm majors now match.** torch ships its own
  `libamdhip64.so` with SONAME `libamdhip64.so.7` and the system has one too. Two copies of the same
  soname in one process make HIP abort:

  ```
  F hip.cpp:512] hipApiName has non-null function pointer 1 despite this being the first
                 instance of the library being copies
  ```

  seen as a hard abort during scene reconfiguration. Matching the ROCm major is what makes the fix
  *safe for torchvision*: once torch's copies are moved aside, the system's `.so.7` satisfies the
  soname `torchvision`'s `_C.so` needs. On a ROCm 6 wheel it could not, and GPU physics and
  torchvision were mutually exclusive. **Match the major AND apply the fix.**

- **torch and torchrl must be bumped together.** torchrl `0.N` is compiled against torch `2.(N-1)`,
  so `torchrl 0.13.x` pairs with **torch 2.12**, not the newest wheel on the index. Get it wrong and
  the failure is nasty: torchrl catches the `ImportError` from its own extension and degrades
  quietly, so rollout collection runs for thousands of steps and the process **segfaults at the
  first gradient update**, when the replay buffer first touches the C++ path. The crash tracks
  `--learning-starts` exactly, which is the tell. Check it directly:

  ```bash
  uv run python -c "import torchrl._torchrl"     # must not raise
  ```

- **torch must come from the ROCm index, matching the system ROCm major.** `cat /opt/rocm/.info/version`;
  7.x means the `rocm7.1` index and torch >= 2.10. `pyproject.toml` handles this. Do not "simplify"
  the `[tool.uv.sources]` block away.

- **`ulimit -n 65535` before launching.** SAPIEN's Vulkan interop spends one file descriptor per
  imported buffer, so 1024 envs x 2 cameras exceeds the usual 1024 soft limit. Vulkan reports it as
  `vk::Device::getMemoryFdKHR: ErrorOutOfDeviceMemory`, which points you at VRAM instead of at
  descriptors. Only bites at high env counts, so small test runs will not catch it.

- **`--no-compile` is required.** torch.compile dies at the first gradient update with
  `InductorError: CUDA driver error: 301` (Triton JIT on ROCm).

- **`--replay-episodes 0.5` on a 48 GB card.** The default 2.0 asks for a 409,600-transition buffer,
  about 69 GB of uint8 at 168x168 with two cameras. That fits a 96 GB card, not a 48 GB one.
  Measured: 0.5 is stable at 108.8 sps; 0.75 and 1.0 both segfault. Note this is a real deviation
  from the NVIDIA baseline: a 4x smaller replay buffer is less sample-efficient, so do not assume
  the same step count reaches the same success rate.

- **`WANDB_MODE=offline`, then `wandb sync`.** Online W&B kills the process. Established by A/B/A:
  `--no-track` passes, `--track` segfaults, `--no-track` passes again. Its system-metrics collector
  polls the GPU and disturbs the HIP context. Offline keeps every metric.

- **`squint` cannot train on this hardware.** Conv backward at batch >= 256 crashes 4 runs in 6 on
  gfx1100 + ROCm 7.2.1. It reproduces in ~25 lines of pure PyTorch with no simulator and on two
  different torch versions, so it is a ROCm bug rather than anything in this project or the port.
  `dino_patch` is unaffected: its DINOv2 backbone is frozen, so no convolution appears in the
  backward pass. On AMD, that is a reason to prefer the ViT encoder rather than a stylistic choice.

- **DINOv2 weights need a warm `torch.hub` cache if GitHub is unreachable.** On the AMD cloud box
  `github.com` is blocked while the weights CDN is not. Fetch the repo through a proxy into
  `~/.cache/torch/hub/facebookresearch_dinov2_main` and the checkpoint into
  `~/.cache/torch/hub/checkpoints`, and it loads offline.

- **Rendering works, but its synchronisation is host-side on AMD**, because ROCm cannot import Vulkan
  timeline semaphores. Render and compute do not overlap, so visual training is slower than the
  ~1.67× physics gap alone would predict. Measure it rather than extrapolating.

- **Requires RDNA3 (gfx1100).** CDNA (MI200/MI300) is wave64 and unsupported. The port's build
  refuses it rather than produce silently wrong physics.

## What is verified

The port's physics is validated in detail: 15/15 checks, bit-identical determinism, GPU-vs-CPU
divergence matching NVIDIA to three significant figures, and a policy trained on AMD scoring 98.4%
on NVIDIA.

On this branch specifically:

| | |
|---|---|
| Environment resolves and installs on AMD, `mani_skill` unmodified | verified |
| GPU physics, INV-P1 confirms the GPU is genuinely in use | verified |
| `obs_mode="rgb"` returns real frames on `cuda:0` | verified |
| Scene reconfiguration is stable (the render interop survives it) | verified |
| Rollout collection, 64 envs | verified, ~562 it/s |
| SAC gradient updates | see below |
| Training to convergence | not yet run |
| Deploying an AMD-trained policy to the physical rig | not yet run |

Treat a long run as the first of its kind and watch the first few thousand steps.
