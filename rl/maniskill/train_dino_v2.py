"""Squint SAC with a purpose-built DINOv2 perception head (v2).

Keeps the SAC training machinery (off-policy, autotuned entropy, C51 distributional critic
ensemble, asymmetric privileged critic, layer norm, torch.compile) but redesigns the vision
path to actually exploit a frozen DINOv2 ViT-S/14 (registers) backbone, instead of the
squint-CNN-shaped head in `train_dino.py`:

  - `DinoEncoder`: frozen DINOv2 patch tokens per camera + learned camera embeddings + K
    learned query tokens that CROSS-ATTEND the token set (attention pooling) -> a 512-d
    visual vector. This keeps the spatial selectivity a conv-collapse discards.
  - `Projection`: RGB projected to 256 (not squint's 50), so the pooled features aren't
    strangled through a narrow bottleneck before fusing with proprioception.

Everything else (env, DR, reward, SAC loop) is unchanged from train_squint/train_dino.
Obs is 112px (a multiple of the 14px patch -> an 8x8 token grid per camera).

SPEED CAVEAT (same as train_dino.py, unchanged here): the frozen ViT runs on every replay
batch (no feature caching), so throughput is a few hundred env-steps/s. `torch.compile` is
ON (tolerates the ViT's graph breaks); `cudagraphs` is OFF (the ViT's positional-embedding
interpolation breaks static-shape capture). The next optimization is to cache the frozen
backbone's patch tokens in the replay buffer and train only the head on them (run the ViT
once per env-step at rollout); that's a buffer restructure left as a fast-follow so the
architecture change here stays isolated and A/B-able against train_dino.py. First run
downloads DINOv2 weights via torch.hub (needs internet + ~/.cache/torch/hub).

Usage (from rl/maniskill/, after `uv sync`):
    uv run python train_dino_v2.py --env_id=SOFramePickPlaceBin-v1

Vendored from the paper's reference implementation
(https://github.com/aalmuzairee/squint/blob/main/train_squint.py), Almuzairee & Christensen,
2026 (arxiv.org/abs/2602.21203). Local changes on top of upstream:

- registers this repo's SO-101-on-frame agent and `SOFramePickPlaceBin-v1` task, with
  matching default `env_id`/`wandb_project_name`
- warm starts: `--checkpoint` runs collect with the loaded policy before learning starts,
  and `--reset_alpha` (default on) restores exploration
- env flags: `--randomize_colors`, `--action_rate_penalty`, `--visual_fidelity`
- tracks the best eval checkpoint (ckpt_best.pt) separately from the latest one

Usage (from rl/maniskill/, after `uv sync`):
    uv run python train_squint.py --env_id=SOFramePickPlaceBin-v1
"""

import os
os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["EXCLUDE_TD_FROM_PYTREE"] = "1"
os.environ["TORCH_LOGS"] = "-dynamo,-inductor"

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message="Using lock_\\(\\) in a compiled graph")

from collections import defaultdict, deque
from dataclasses import dataclass

import math
import random
import time
import glob
from typing import Optional

from mani_skill.utils import common as ms_common
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import tqdm
import wandb
from tensordict import TensorDict, from_module, from_modules
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

# Add tasks: registers the SO-101-on-frame agent and SOFramePickPlaceBin-v1 task.
import soframe_rl_maniskill.envs  # noqa: F401
import mani_skill.envs  # noqa: F401

import utils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass
class Args:
    exp_name: Optional[str] = "baseline"
    """the name of this experiment"""
    agent_name: Optional[str] = "squint"
    """for logging and tracking"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_project_name: str = "so-frame-maniskill"
    """the wandb's project name"""
    wandb_group: str = "SQUINT"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_trajectory: bool = False
    """whether to save trajectory data into the `videos` folder"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder, and to wandb if wandb is set"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""
    reset_alpha: bool = True
    """when warm-starting from --checkpoint, keep the default entropy temperature instead
    of the checkpoint's collapsed one (~1e-4 after long training, which leaves the
    fine-tuning run no exploration); pass --no-reset_alpha to inherit it"""

    # Environment specific arguments
    env_id: str = "SOFramePickPlaceBin-v1"
    """the id of the environment"""
    env_domain_randomization: bool = True
    """adds domain randomization flag if env supports it"""
    randomize_colors: bool = False
    """also randomize the bar and bin colors per scene build (they default to fixed
    purple/yellow matched to real captures); pair with --reconfiguration_freq so
    training re-samples them"""
    overhead_camera_fov: Optional[float] = None
    """override the overhead camera's base FOV, in DEGREES (default: the URDF-derived
    value in envs/base_random_env.py). Set to the measured real-camera FOV reported by
    examples/calibrate_camera.py as fov_deg."""
    wrist_camera_fov: Optional[float] = None
    """override the wrist camera's base FOV, in DEGREES; same sourcing as overhead's"""
    overhead_camera_pos_offset: Optional[tuple[float, float, float]] = None
    """constant position correction for the overhead camera, meters in the camera link's
    local frame (+X = view direction). Calibrated against a rectified real frame, see
    examples/calibrate_camera.py"""
    overhead_camera_rot_offset: Optional[tuple[float, float, float]] = None
    """constant rotation correction for the overhead camera, roll/pitch/yaw in DEGREES
    (camera-local); same sourcing as the position offset"""
    action_rate_penalty: float = 0.0
    """smoothness cost: -k * ||a_t - a_{t-1}||^2 per step (raw reward units). Penalizes
    jerk, not movement. ~0.05 is a reasonable strength for polish runs."""
    visual_fidelity: str = "raster"
    """rendering fidelity, see `RandomizationConfig.visual_fidelity` in
    envs/base_random_env.py. "raster" (default) trains on realistic appearance (PBR
    materials + softbox-like lighting with a faint shadow); "flat" is the fast
    shadowless look for cheap ablations; "raytraced" applies only to eval_envs,
    rendering their wrist/overhead sensor cameras with ray tracing on the cpu sim
    backend -- combine with `--evaluate --checkpoint <path> --num_eval_envs 1` for a
    single high-fidelity rollout."""
    num_envs: int = 512
    """the number of parallel environments (dino: lowered from 1024; 112px obs + a ViT are memory-heavy)"""
    num_eval_envs: int = 32
    """the number of parallel evaluation environments"""
    partial_reset: bool = False
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    eval_freq: int = 100_000
    """evaluation frequency in terms of global steps"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment"""
    obs_mode: Optional[str] = "rgb+segmentation"
    """the observation output mode of the environment"""
    privileged_critic: bool = True
    """asymmetric actor-critic: append '+state' to the obs_mode so the env emits its
    privileged ground truth (_get_obs_extra: item/bin/tcp poses, randomized physics
    params), split it out of the flattened state, and feed it to the CRITIC only. The
    actor (and deployed policy) keeps vision + proprio. Speeds up value learning with
    zero sim2real cost since the critic never runs on the robot."""
    action_delay_max: Optional[int] = None
    """override the env's max action delay (control steps). Set 0 to disable the
    delay-obs augmentation entirely, which keeps the proprio state at its base 14 dims
    -- needed to cleanly warm-start a pre-delay checkpoint (e.g. model_best_raster.pt).
    Default None uses the env config's range. (The task is quasi-static, so the deployed
    recipe leaves delay OFF and relies on the tight 10 Hz control loop instead.)"""
    arm_speed_scale: float = 1.0
    """multiplier on arm/rail delta limits (gripper unscaled). 1.0 = calibrated slow arm
    (0.5 rad/s, 0.12 m/s); 2.0 = old fast v24/v25 speeds. For arm-speed ablations."""
    render_mode: Optional[str] = "all"
    """the rendering mode of the environment, could be rgb or all"""
    render_size: int = 224
    """square size to render from env (HxW) - before downsampling (dino: 224 -> squint 112 keeps antialiasing)"""
    image_size: int = 112
    """square size of the input image for actor (HxW) - after downsampling (dino: 112 = 8x the 14px patch)"""
    apply_jitter: bool = True
    """applies color jitter to all input RGB observations (better for sim2real)"""
    sensor_aug: bool = True
    """applies camera sensor-realism augmentation (h264/compression proxy, sensor noise,
    gamma/white-balance) on top of color jitter -- closes the clean-render vs real-camera
    gap. See utils.SensorAugWrapper."""

    # Algorithm specific arguments
    total_timesteps: int = 1_500_000
    """total timesteps of the experiments"""
    buffer_size: int = 200_000
    """the replay memory buffer size (dino: lowered from 1M; 112px images are ~75 KB each)"""
    batch_size: int = 256
    """the batch size of sample from the replay memory (dino: lowered from 512 for the ViT forward)"""
    num_updates: int = 32
    """num updates per parallel env step (dino: lowered from 256 to keep ~UTD at num_envs=512, and because
    each update runs the frozen ViT on the batch, no feature caching yet, so this dominates step time)"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha for policy"""
    policy_frequency: int = 4
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    bootstrap_at_done: str = "always"
    """bootstrap method when episode ends. Options: ['always', 'never', 'on_truncation']"""
    gamma: float = 0.9
    """the discount factor gamma"""
    tau: float = 0.01
    """target smoothing coefficient"""
    num_q: int = 2
    """number of Q-networks in the critic ensemble"""
    num_atoms: int = 101
    """number of atoms for distributional RL (C51)"""
    v_min: float = -20.0
    """minimum value for distributional RL support"""
    v_max: float = 20.0
    """maximum value for distributional RL support"""

    # Optimizations
    compile: bool = True
    """whether to use torch.compile. (dino: ON; tolerant of the ViT's graph breaks, falls back to eager
    around them, so it still speeds up the MLP actor/C51 critic)"""
    cudagraphs: bool = False
    """whether to use cudagraphs on top of compile. (dino: OFF; unlike compile it needs static shapes /
    no dynamic control flow, which the torch.hub ViT's positional-embedding interpolation breaks)"""

    # to be filled in runtime
    num_total_iterations: int = 0
    """the number of parallel envs steps given global total timesteps"""


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(args, eval_envs, get_action_fn, logger, eval_output_dir, max_episode_steps, global_step, pbar):
    torch.cuda.empty_cache()
    stime = time.perf_counter()
    eval_obs, _ = eval_envs.reset()
    eval_metrics = defaultdict(list)

    for _ in range(max_episode_steps):
        with torch.no_grad():
            eval_action = get_action_fn(eval_obs['rgb'], eval_obs['state'])
            eval_obs, _, _, _, eval_infos = eval_envs.step(eval_action)
            if "final_info" in eval_infos:
                mask = eval_infos["_final_info"]
                for k, v in eval_infos["final_info"]["episode"].items():
                    eval_metrics[f'eval/{k}'].append(v[mask])

    eval_d = {}
    for k, v in eval_metrics.items():
        eval_d[k] = torch.stack(v).float().mean()

    pbar.set_description(
        f"success_at_end: {eval_d['eval/success_at_end']:.2f}, "
        f"success_once: {eval_d['eval/success_once']:.2f}, "
        f"return: {eval_d['eval/return']:.2f}"
    )
    eval_time = time.perf_counter() - stime
    eval_d["time/eval_time"] = eval_time

    if args.track and args.capture_video:
        video_files = glob.glob(f"{eval_output_dir}/*.mp4")
        if video_files:
            latest_video = max(video_files, key=os.path.getctime)
            eval_d["eval/video"] = wandb.Video(latest_video, format="mp4")

    logger.total_eval_time += eval_time
    logger.log(d=eval_d, step=global_step)
    return eval_d


# ─────────────────────────────────────────────────────────────────────────────
#  Network Modules
# ─────────────────────────────────────────────────────────────────────────────

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class CNNEncoder(nn.Module):
    def __init__(self, n_obs, device=None):
        super().__init__()
        assert len(n_obs) == 3 and n_obs[0] == n_obs[1]
        self.num_channels = n_obs[2]
        self.image_size = n_obs[0]
        self.repr_dim = 1024

        if self.image_size == 64:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 8, stride=4, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif self.image_size == 32:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        elif self.image_size == 16:
            self.conv = nn.Sequential(
                nn.Conv2d(self.num_channels, 32, 4, stride=2, device=device), nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=1, device=device), nn.ReLU(),
                nn.Flatten()
            )
        else:
            raise ValueError(f"No CNN encoder supported for image size: {self.image_size}")

        self.apply(weight_init)
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        obs = obs.contiguous(memory_format=torch.channels_last)
        obs = obs / 255.0 - 0.5
        return self.conv(obs)


# ── DINOv2 frozen-backbone encoder ──────────────────────────────────────────
# The ViT is loaded once and shared across the encoder copies (it's frozen and
# identical everywhere; only the small conv head differs and is weight-shared by
# from_module). Cache keyed by device string.
_DINO_BACKBONE = {}


def _load_dino_backbone(device):
    key = str(device)
    if key not in _DINO_BACKBONE:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _DINO_BACKBONE[key] = m
    return _DINO_BACKBONE[key]


class DinoEncoder(nn.Module):
    """Frozen DINOv2 ViT-S/14 (registers) backbone + an ATTENTION-POOLING head.

    Drop-in for CNNEncoder (`__init__(n_obs, device)`, `.repr_dim`,
    `forward(rgb_uint8[B,H,W,C]) -> [B, repr_dim]`), but it treats DINOv2's output as a
    SET OF TOKENS rather than a feature image:
      - each camera's 3 channels -> DINOv2 patch tokens, [B, grid*grid, EMBED];
      - a learned per-camera embedding is added, and the cameras' token sets are
        concatenated into one sequence, [B, num_cams*grid*grid, EMBED];
      - K learned query tokens cross-attend that sequence (nn.MultiheadAttention), so the
        head LEARNS WHERE TO LOOK (bar / bin / gripper) instead of conv-crushing the grid;
      - the K pooled vectors are projected to repr_dim (512).
    This keeps the spatial selectivity a conv-collapse throws away, and repr_dim is wide
    (512) so the downstream Projection no longer strangles the visual features (see below).

    The frozen backbone is kept OUT of parameters()/state_dict() (object.__setattr__), so
    only the head (camera embeddings, queries, attention, projection) trains, checkpoints
    stay small, and the from_module weight-sharing across encoder copies shares just the head.
    """

    PATCH = 14
    EMBED = 384  # DINOv2 ViT-S

    def __init__(self, n_obs, device=None, n_queries=8, n_heads=6, repr_dim=512):
        super().__init__()
        assert len(n_obs) == 3 and n_obs[0] == n_obs[1], "square obs expected"
        assert n_obs[2] % 3 == 0, "expects stacked 3-channel cameras"
        self.image_size = n_obs[0]
        self.num_cams = n_obs[2] // 3
        # DINOv2 needs a side that is a multiple of the 14px patch.
        self.dino_res = max(2 * self.PATCH, (self.image_size // self.PATCH) * self.PATCH)
        self.grid = self.dino_res // self.PATCH
        self.tokens_per_cam = self.grid * self.grid
        self.repr_dim = repr_dim

        object.__setattr__(self, "_dino", _load_dino_backbone(device))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

        # Trainable head: camera-id embeddings + learned query tokens + cross-attention.
        self.cam_embed = nn.Parameter(torch.zeros(self.num_cams, self.EMBED, device=device))
        self.queries = nn.Parameter(torch.zeros(n_queries, self.EMBED, device=device))
        self.ln_kv = nn.LayerNorm(self.EMBED, device=device)
        self.ln_q = nn.LayerNorm(self.EMBED, device=device)
        self.attn = nn.MultiheadAttention(self.EMBED, n_heads, batch_first=True, device=device)
        self.out = nn.Sequential(
            nn.Linear(n_queries * self.EMBED, repr_dim, device=device),
            nn.LayerNorm(repr_dim, device=device), nn.ReLU(),
        )
        self.apply(weight_init)
        # Raw Parameters are untouched by weight_init (it only hits Linear/Conv) -- init them here.
        nn.init.normal_(self.queries, std=0.02)
        nn.init.zeros_(self.cam_embed)

    def _patch_tokens(self, img3):
        # img3: [B, 3, dino_res, dino_res] normalized -> [B, tokens_per_cam, EMBED]
        return self._dino.forward_features(img3)["x_norm_patchtokens"]

    def forward(self, obs):
        # obs: [B, H, W, C] uint8, C = 3 * num_cams
        x = obs.permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():  # frozen backbone: no graph, no grad
            toks = []
            for c in range(self.num_cams):
                cam = x[:, 3 * c: 3 * c + 3]
                if cam.shape[-1] != self.dino_res:
                    cam = F.interpolate(cam, size=(self.dino_res, self.dino_res),
                                        mode="bilinear", align_corners=False)
                cam = (cam - self.mean) / self.std
                toks.append(self._patch_tokens(cam))
        # Trainable head (gradients flow here, not into the backbone).
        toks = [t + self.cam_embed[c] for c, t in enumerate(toks)]   # tag each token's camera
        kv = self.ln_kv(torch.cat(toks, dim=1))                      # [B, n_tok, EMBED]
        b = kv.shape[0]
        q = self.ln_q(self.queries).unsqueeze(0).expand(b, -1, -1)   # [B, K, EMBED]
        pooled, _ = self.attn(q, kv, kv, need_weights=False)         # [B, K, EMBED]
        return self.out(pooled.reshape(b, -1))                       # [B, repr_dim]


class Projection(nn.Module):
    # v2: RGB projected to 256 (not squint's 50) so the attention-pooled DINOv2 features
    # aren't strangled through a narrow bottleneck before fusing with proprioception.
    def __init__(self, n_obs, n_state, device=None):
        super().__init__()
        self.repr_dim = 256 + 256
        self.rgb_proj = nn.Sequential(
            nn.Linear(n_obs, 256, device=device), nn.LayerNorm(256, device=device), nn.Tanh(),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(n_state, 256, device=device), nn.LayerNorm(256, device=device), nn.ReLU(),
        )

    def forward(self, rgb, state):
        return torch.cat([self.rgb_proj(rgb), self.state_proj(state)], dim=-1)


class Actor(nn.Module):
    def __init__(self, env, n_obs, n_state, n_act, device=None):
        super().__init__()
        hidden_dim = 256
        activ = nn.ReLU

        self.proj = Projection(n_obs, n_state, device=device)
        self.fc = nn.Sequential(
            nn.Linear(self.proj.repr_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), activ(),
        )
        self.fc_mean = nn.Linear(hidden_dim, n_act, device=device)
        self.fc_logstd = nn.Linear(hidden_dim, n_act, device=device)

        action_space = env.unwrapped.single_action_space
        self.register_buffer("action_scale",
            torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32, device=device))
        self.register_buffer("action_bias",
            torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32, device=device))

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -5
        self.apply(weight_init)

    def forward(self, rgb, state, get_log_std=False):
        x = self.proj(rgb, state)
        x = self.fc(x)
        mean = self.fc_mean(x)
        if get_log_std:
            log_std = self.fc_logstd(x)
            log_std = torch.tanh(log_std)
            log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)
            return mean, log_std
        return mean

    def get_eval_action(self, rgb, state):
        mean = self.forward(rgb, state)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, rgb, state):
        mean, log_std = self.forward(rgb, state, get_log_std=True)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing action bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class Critic(nn.Module):
    """Distributional C51 Ensemble-Q-network critic with vmap optimizations."""
    def __init__(self, n_obs, n_state, n_act, num_atoms, v_min, v_max, num_q=2, device=None):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_q = num_q
        self.v_min = v_min
        self.v_max = v_max
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=device)

        self.proj = Projection(n_obs, n_state, device=device)
        self.proj.apply(weight_init)

        q_input_dim = self.proj.repr_dim + n_act

        # Build Q-networks, apply weight init, then stack into q_params
        q_nets = [self._build_q_network(q_input_dim, num_atoms, device=device) for _ in range(num_q)]
        for qn in q_nets:
            qn.apply(weight_init)

        # q_params: registered stacked parameter container (what optimizer + vmap both use)
        self.q_params = from_modules(*q_nets, as_module=True)

        # Meta-device template for vmap dispatch (hidden from parameters()/state_dict())
        object.__setattr__(self, '_q_meta', self._build_q_network(q_input_dim, num_atoms, device="meta"))

        # Store architecture string for __repr__
        object.__setattr__(self, '_q_repr', repr(q_nets[0]))

    def __repr__(self):
        """Pretty module printing"""
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"  (proj): {self.proj}")
        for i in range(self.num_q):
            lines.append(f"  (q{i}): {self._q_repr}")
        lines.append(")")
        return "\n".join(lines)

    def _build_q_network(self, input_dim, num_atoms, device=None):
        """Build a single Q-network. Used for q_nets, meta template."""
        hidden_dim = 512
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=device), nn.LayerNorm(hidden_dim, device=device), nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms, device=device)
        )

    def _vmap_q(self, params, x):
        """Single Q-network forward through meta template. Dispatched by vmap."""
        with params.to_module(self._q_meta):
            return self._q_meta(x)

    def forward(self, rgb_features, state, actions):
        """Batched forward: [num_q, batch, num_atoms]. Full gradient flow through all params."""
        proj = self.proj(rgb_features, state)
        x = torch.cat([proj, actions], dim=-1)
        return torch.vmap(self._vmap_q, (0, None))(self.q_params, x)

    def get_q_values(self, rgb_features, state, actions, detach_critic=False):
        """Expected Q-values: [num_q, batch].

        Args:
            detach_critic: If True, freezes critic weights (proj + Q-networks) while
                preserving gradients through actions. Used for actor policy gradient.
        """
        if detach_critic:
            with torch.no_grad():
                proj = self.proj(rgb_features, state)
            x = torch.cat([proj, actions], dim=-1)
            logits = torch.vmap(self._vmap_q, (0, None))(self.q_params.data, x)
        else:
            logits = self.forward(rgb_features, state, actions)
        probs = F.softmax(logits, dim=-1)
        return torch.sum(probs * self.q_support, dim=-1)

    def categorical(self, rgb_features, state, actions, rewards, bootstrap, discount):
        """C51 categorical projection: [num_q, batch, num_atoms].
        Called under no_grad for target computation."""
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]
        device = rewards.device

        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount * self.q_support
        target_z = target_z.clamp(self.v_min, self.v_max)

        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        is_integer = upper == lower
        lower = torch.where(torch.logical_and(lower > 0, is_integer), lower - 1, lower)
        upper = torch.where(torch.logical_and(lower == 0, is_integer), upper + 1, upper)

        # Batched forward through all Q-networks via vmap
        logits = self.forward(rgb_features, state, actions)  # [num_q, batch, atoms]
        next_dists = F.softmax(logits, dim=-1)

        # Fused projection: reshape to [num_q*batch, atoms]
        total_batch = self.num_q * batch_size
        next_dists_flat = next_dists.reshape(-1, self.num_atoms)
        offset = torch.arange(total_batch, device=device).unsqueeze(1) * self.num_atoms

        lower_exp = lower.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        upper_exp = upper.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)
        b_exp = b.unsqueeze(0).expand(self.num_q, -1, -1).reshape(total_batch, self.num_atoms)

        max_index = total_batch * self.num_atoms - 1
        lower_indices = torch.clamp((lower_exp + offset).view(-1), 0, max_index)
        upper_indices = torch.clamp((upper_exp + offset).view(-1), 0, max_index)

        proj_dist_flat = torch.zeros_like(next_dists_flat)
        proj_dist_flat.view(-1).index_add_(0, lower_indices, (next_dists_flat * (upper_exp.float() - b_exp)).view(-1))
        proj_dist_flat.view(-1).index_add_(0, upper_indices, (next_dists_flat * (b_exp - lower_exp.float())).view(-1))

        return proj_dist_flat.reshape(self.num_q, batch_size, self.num_atoms)


# ─────────────────────────────────────────────────────────────────────────────
#  Deployment Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class DeployAgent(nn.Module):
    """Standalone deployment wrapper for a real-robot deploy script. Handles downsampling and inference."""

    def __init__(self, sim_env, sample_obs, target_image_size=32, device=None):
        super().__init__()
        self.device = device
        self.target_image_size = target_image_size

        n_act = np.prod(sim_env.unwrapped.single_action_space.shape)
        n_obs_shape = sample_obs['rgb'].shape
        c = n_obs_shape[3] if len(n_obs_shape) == 4 else n_obs_shape[2]
        n_obs = (target_image_size, target_image_size, c)
        n_state = np.prod(sample_obs['state'].shape[1:]) if len(sample_obs['state'].shape) > 1 else sample_obs['state'].shape[0]

        self.encoder = DinoEncoder(n_obs, device)
        self.actor = Actor(sim_env, n_obs=self.encoder.repr_dim, n_state=n_state, n_act=n_act, device=self.device)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint, map_location=self.device)
        self.encoder.load_state_dict(ckpt['encoder'])
        self.actor.load_state_dict(ckpt['actor'])
        print(f"Loaded checkpoint from {checkpoint} at step {ckpt['global_step']}")

    def downsample_rgb(self, rgb):
        if rgb.shape[-3] == self.target_image_size:
            return rgb
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.permute(0, 3, 1, 2).float()
        rgb = F.interpolate(rgb, size=(self.target_image_size, self.target_image_size), mode='area')
        rgb = rgb.permute(0, 2, 3, 1).to(torch.uint8)
        if squeeze:
            rgb = rgb.squeeze(0)
        return rgb

    def get_action(self, obs):
        rgb = self.downsample_rgb(obs['rgb'])
        with torch.no_grad():
            rgb = self.encoder(rgb)
            return self.actor.get_eval_action(rgb, obs['state'])

    def forward(self, obs):
        return self.get_action(obs)


# ─────────────────────────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_wandb=False):
        self.log_wandb = log_wandb
        self.start_time = time.perf_counter()
        self.total_eval_time = 0 # to subtract from total wall_time

    @property
    def wall_time(self):
        return time.perf_counter() - self.start_time - self.total_eval_time

    def log(self, d, step):
        if self.log_wandb:
            d["time/wall_time"] = self.wall_time
            wandb.log(d, step=step)

    def close(self):
        if self.log_wandb:
            wandb.finish()

    def upload_checkpoint(self, model_path: str, model_name="model_checkpoint"):
        if self.log_wandb:
            artifact = wandb.Artifact(name=model_name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            artifact.wait()
            print(f"Uploaded checkpoint {model_name} to wandb")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.num_total_iterations = int(args.total_timesteps // args.num_envs)
    assert args.num_updates > 0, "No updates will be made to the model with the current setup"

    # Arm-speed ablation knob: set before any env/agent is built so the controller picks
    # it up when its config is constructed.
    import soframe_rl_maniskill.robot.so101_on_frame as _robot_mod
    _robot_mod.ARM_SPEED_SCALE = args.arm_speed_scale
    assert args.visual_fidelity in ("flat", "raster", "raytraced"), \
        f"--visual_fidelity must be flat|raster|raytraced, got {args.visual_fidelity!r}"

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name
    model_path = os.path.abspath(f"runs/{run_name}/ckpt.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # ── Environment setup ──────────────────────────────────────────────────
    obs_mode = args.obs_mode
    if args.privileged_critic and "state" not in obs_mode.split("+"):
        # '+state' makes the env emit _get_obs_extra's privileged ground truth, which
        # SplitPrivilegedStateWrapper below routes to the critic only.
        obs_mode = obs_mode + "+state"
    env_kwargs = dict(obs_mode=obs_mode, render_mode=args.render_mode, sim_backend="gpu",
                      sensor_configs=dict(width=args.render_size, height=args.render_size))
    # "sensors" (wrist + overhead only, no third-person camera) for whatever eval_envs
    # records -- both the periodic in-training eval videos and --evaluate rollouts.
    eval_env_kwargs = dict(obs_mode=obs_mode, render_mode="sensors", sim_backend="gpu",
                           sensor_configs=dict(width=args.render_size, height=args.render_size),
                           human_render_camera_configs=dict(shader_pack="default", width=args.render_size, height=args.render_size))
    for kw in (env_kwargs, eval_env_kwargs):
        if args.control_mode is not None:
            kw["control_mode"] = args.control_mode
        if args.env_domain_randomization:
            kw["domain_randomization"] = True
        if args.action_rate_penalty > 0:
            kw["action_rate_penalty"] = args.action_rate_penalty
        dr_cfg = {}
        if args.randomize_colors:
            dr_cfg.update(randomize_item_color=True, randomize_bin_color=True)
        if args.overhead_camera_fov is not None:
            dr_cfg["overhead_camera_fov"] = np.deg2rad(args.overhead_camera_fov)
        if args.wrist_camera_fov is not None:
            dr_cfg["wrist_camera_fov"] = np.deg2rad(args.wrist_camera_fov)
        if args.overhead_camera_pos_offset is not None:
            dr_cfg["overhead_camera_pos_offset"] = list(args.overhead_camera_pos_offset)
        if args.overhead_camera_rot_offset is not None:
            dr_cfg["overhead_camera_rot_offset"] = [np.deg2rad(v) for v in args.overhead_camera_rot_offset]
        if args.action_delay_max is not None:
            dr_cfg["action_delay_steps_range"] = (0, args.action_delay_max)
        if args.visual_fidelity in ("flat", "raster"):
            # Always pass the flag through so --visual_fidelity flat can override the
            # config's raster default. No shader change needed for raster: the
            # memory-optimized "minimal" sensor shader renders shadows and textures
            # just fine (verified), while the "default" shader's G-buffers OOM the GPU
            # beyond ~384 envs for no visible benefit.
            dr_cfg["visual_fidelity"] = args.visual_fidelity
        if dr_cfg:
            kw["domain_randomization_config"] = dr_cfg
    if args.visual_fidelity == "raytraced":
        # Ray-traced sensor cameras only exist on the cpu sim backend (single env; the
        # GPU-parallel camera path is rasterizer-only), so both env pools drop to cpu
        # with rt-fast sensors. For --evaluate that's the usual one-off render path.
        # For TRAINING it means slow single-env collection (~190 steps/s at 128 px):
        # viable for short visual fine-tunes, not from-scratch runs -- and
        # --num_updates must be scaled down to keep the update-to-data ratio sane
        # (the default 256 per iteration assumes 1024 env steps per iteration).
        for kw in (env_kwargs, eval_env_kwargs):
            kw["sensor_configs"]["shader_pack"] = "rt-fast"
            kw["domain_randomization_config"] = dict(
                kw.get("domain_randomization_config", {}), visual_fidelity="raytraced"
            )
            kw["sim_backend"] = "cpu"
        if not args.evaluate:
            assert args.num_envs == 1 and args.num_eval_envs == 1, (
                "raytraced training runs on the cpu backend, which supports a single "
                "env: pass --num_envs=1 --num_eval_envs=1 (and a small --num_updates)"
            )

    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq, **eval_env_kwargs)
    max_episode_steps = gym_utils.find_max_episode_steps_value(envs)

    # The env's flattened 'state' is [agent_flat | extra_flat] in that order (ManiSkill
    # builds dict(agent=..., extra=...) then flattens by insertion order). The agent
    # (proprio) size is the split point between actor-visible proprio and critic-only
    # privileged state; measure it from _get_obs_agent so it's robust to how the env
    # packs its obs dict, then derive priv by difference against the merged vector.
    n_priv = 0
    if args.privileged_critic:
        agent_obs = envs.unwrapped._get_obs_agent()
        n_proprio = int(ms_common.flatten_state_dict(agent_obs, use_torch=True).shape[-1])
        n_merged = int(envs.unwrapped._init_raw_obs["state"].shape[-1])
        n_priv = n_merged - n_proprio
        assert n_priv > 0, f"expected privileged state beyond proprio (merged={n_merged}, proprio={n_proprio})"

    envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=True)
    eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=True)
    if args.privileged_critic:
        envs = utils.SplitPrivilegedStateWrapper(envs, n_proprio)
        eval_envs = utils.SplitPrivilegedStateWrapper(eval_envs, n_proprio)

    if args.render_size != args.image_size:
        envs = utils.DownsampleObsWrapper(envs, target_size=args.image_size)
        eval_envs = utils.DownsampleObsWrapper(eval_envs, target_size=args.image_size)
    if args.apply_jitter:
        envs = utils.ColorJitterWrapper(envs)
        eval_envs = utils.ColorJitterWrapper(eval_envs)
    if args.sensor_aug:
        # After ColorJitter: adds compression/noise/exposure realism. Applied to eval
        # too (like ColorJitter) so the best checkpoint is chosen for robustness, not
        # clean-image performance -- expect success ~a bit below a no-aug run.
        envs = utils.SensorAugWrapper(envs)
        eval_envs = utils.SensorAugWrapper(eval_envs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    eval_output_dir = None
    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"runs/{run_name}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // max_episode_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False,
                                 save_video_trigger=save_video_trigger, max_steps_per_video=max_episode_steps, video_fps=10)  # 10 Hz control -> real-time playback
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.save_trajectory,
                                  save_video=args.capture_video, trajectory_name="trajectory",
                                  max_steps_per_video=max_episode_steps, video_fps=10)  # 10 Hz control -> real-time playback

    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)

    n_act = math.prod(envs.unwrapped.single_action_space.shape)
    n_channels = envs.unwrapped.single_observation_space['rgb'].shape[2]
    n_obs = (args.image_size, args.image_size, n_channels)
    n_state = math.prod(envs.unwrapped.single_observation_space['state'].shape)
    # Critic sees proprio + privileged (asymmetric); actor sees proprio only.
    n_state_critic = n_state + n_priv
    assert isinstance(envs.unwrapped.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    # ── Logger ─────────────────────────────────────────────────────────────
    if not args.evaluate:
        print("Running training")
        if args.track:
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id,
                                     reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**eval_env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id,
                                          reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.eval_partial_reset)
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=False,
                       config=config, name=run_name, save_code=True, group=args.wandb_group,
                       tags=[args.wandb_group, args.agent_name, args.env_id, f"seed={args.seed}"])
    else:
        print("Running evaluation")
    logger = Logger(log_wandb=(args.track and not args.evaluate))

    # ── Instantiate modules ────────────────────────────────────────────────

    encoder = DinoEncoder(n_obs=n_obs, device=device)
    actor = Actor(envs, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device)
    critic = Critic(n_obs=encoder.repr_dim, n_state=n_state_critic, n_act=n_act,
                    num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
                    num_q=args.num_q, device=device)

    # Entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.unwrapped.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr, capturable=args.cudagraphs and not args.compile)
    else:
        alpha = torch.as_tensor(args.alpha, device=device)

    # Load checkpoint
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        actor.load_state_dict(ckpt['actor'])
        critic.load_state_dict(ckpt['critic'])
        if 'log_alpha' in ckpt and not args.reset_alpha:
            with torch.no_grad():
                log_alpha.copy_(ckpt['log_alpha'])
                alpha.copy_(log_alpha.exp())
        print(f"Loaded checkpoint from {args.checkpoint} at step {ckpt['global_step']}"
              + (" (log_alpha reset to default for fresh exploration)" if args.reset_alpha else ""))

    # ── Inference copies (weight-sharing via from_module) ──────────────────

    encoder_detach = DinoEncoder(n_obs=n_obs, device=device)
    encoder_eval = DinoEncoder(n_obs=n_obs, device=device).eval()
    from_module(encoder).data.to_module(encoder_detach)
    from_module(encoder).data.to_module(encoder_eval)

    actor_detach = Actor(envs, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device)
    actor_eval = Actor(envs, n_obs=encoder.repr_dim, n_state=n_state, n_act=n_act, device=device).eval()
    from_module(actor).data.to_module(actor_detach)
    from_module(actor).data.to_module(actor_eval)

    # Target critic
    critic_target = Critic(n_obs=encoder.repr_dim, n_state=n_state_critic, n_act=n_act,
                           num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
                           num_q=args.num_q, device=device)
    critic_target.load_state_dict(critic.state_dict())
    critic_online_params = list(critic.parameters())
    critic_target_params = list(critic_target.parameters())

    # ── Inference functions ────────────────────────────────────────────────

    def get_rollout_action(rgb, state):
        # obs arrive on the sim's device, which is cpu under --visual_fidelity=raytraced
        # (single-env cpu backend) while the networks live on `device`.
        rgb_feat = encoder_detach(rgb.to(device))
        action, _, _ = actor_detach.get_action(rgb_feat, state.to(device))
        return action

    def get_eval_action(rgb, state):
        # --visual_fidelity=raytraced runs eval_envs on the cpu sim backend (ray-traced
        # sensor cameras aren't supported on the gpu-parallelized camera path), so obs may
        # not already be on `device` the way they are for the normal all-gpu case.
        rgb_feat = encoder_eval(rgb.to(device))
        return actor_eval.get_eval_action(rgb_feat, state.to(device))

    # ── Optimizers ─────────────────────────────────────────────────────────

    critic_optimizer = optim.Adam(list(critic.parameters()) + list(encoder.parameters()),
                             lr=args.q_lr, capturable=args.cudagraphs and not args.compile)
    actor_optimizer = optim.Adam(list(actor.parameters()),
                                 lr=args.policy_lr, capturable=args.cudagraphs and not args.compile)

    # ── Replay buffer ──────────────────────────────────────────────────────

    rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    # ── Print summary ──────────────────────────────────────────────────────

    print("-----------------------")
    print(args)
    print("-----------------------")
    print("Squint")
    print("-----------------------")
    for mod in [encoder, actor, critic]:
        print(mod)
    print(f"Task: {args.env_id}, Control mode: {envs.unwrapped._control_mode}")
    print(f"Observations: {n_obs}, State: {n_state}, Actions: {n_act}")
    print(f"Device: {device}")
    print("-----------------------")

    # ── Update functions ───────────────────────────────────────────────────

    def _critic_state(obs_dict):
        # Actor sees proprio ('state'); critic additionally sees privileged 'priv'
        # (asymmetric actor-critic). Without --privileged_critic there is no 'priv' key
        # and the critic state is just proprio.
        state = obs_dict['state']
        priv = obs_dict.get('priv', None)
        return state if priv is None else torch.cat([state, priv], dim=-1)

    def update_main(data):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                next_obs = encoder(data["next_observations"]['rgb'])
                next_state = data["next_observations"]['state']
                next_critic_state = _critic_state(data["next_observations"])
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs, next_state)

                bootstrap = (~data["dones"]).float()
                discount = args.gamma
                rewards = data["rewards"].flatten()

                entropy_bonus = alpha * next_state_log_pi.flatten()
                rewards_with_entropy = rewards - bootstrap.flatten() * discount * entropy_bonus

                target_distributions = critic_target.categorical(
                    next_obs, next_critic_state, next_state_actions,
                    rewards_with_entropy, bootstrap, discount
                )

            obs = encoder(data["observations"]['rgb'])
            state = data["observations"]['state']
            critic_state = _critic_state(data["observations"])

            # Shape: [num_q, batch, num_atoms]
            q_outputs = critic(obs, critic_state, data["actions"])
            q_log_probs = F.log_softmax(q_outputs, dim=-1)

            # Cross-entropy: sum over num_atoms, mean over batch → [num_q]
            q_losses = -torch.sum(target_distributions * q_log_probs, dim=-1).mean(dim=-1)

            # Sum over Q-networks losses
            critic_loss = q_losses.sum()

            # Logging q-value metrics
            with torch.no_grad():
                q_probs = F.softmax(q_outputs, dim=-1)
                q_values = torch.sum(q_probs * critic.q_support, dim=-1)
                q_max = q_values.max()
                q_min = q_values.min()

        # Update critic and encoder
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        if args.autotune:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(obs, state)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

            # Update actor entropy
            alpha_optimizer.zero_grad()
            alpha_loss.backward()
            alpha_optimizer.step()

            alpha.copy_(log_alpha.detach().exp())
        else:
            alpha_loss = torch.tensor(0.0, device=device)

        return TensorDict(critic_loss=critic_loss.detach(), q_max=q_max, q_min=q_min,
                          alpha=alpha.detach(), alpha_loss=alpha_loss.detach(),
                          encoded_rgb=obs.detach())

    def update_actor(data, encoded_rgb):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            state = data["observations"]["state"]
            critic_state = _critic_state(data["observations"])
            obs = encoded_rgb

            pi, log_pi, _ = actor.get_action(obs, state)
            q_values = critic.get_q_values(obs, critic_state, pi, detach_critic=True)

            # Mean (No CDQ)
            critic_value = q_values.mean(dim=0)

            actor_loss = (alpha * log_pi - critic_value).mean()

        # Update actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        return TensorDict(actor_loss=actor_loss.detach())

    # ── Compile & CudaGraphs ──────────────────────────────────────────────

    if args.compile:
        update_main = torch.compile(update_main)
        update_actor = torch.compile(update_actor)
        get_rollout_action = torch.compile(get_rollout_action)
        get_eval_action = torch.compile(get_eval_action)

    if args.cudagraphs:
        update_main = CudaGraphModule(update_main)
        update_actor = CudaGraphModule(update_actor)

    # ── Training loop ──────────────────────────────────────────────────────

    obs, _ = envs.reset(seed=args.seed)
    eval_envs.reset(seed=args.seed)

    global_step = 0
    # Where the sim expects actions to live (cpu backend under raytraced training).
    sim_device = torch.device("cpu") if args.visual_fidelity == "raytraced" else device
    pbar = tqdm.tqdm(total=args.total_timesteps, desc="steps")
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    d = {}

    best_eval = (-float("inf"), -float("inf"))  # (success_at_end, return)
    best_model_path = model_path.replace("ckpt.pt", "ckpt_best.pt")

    for iteration in range(args.num_total_iterations + 2):  # +2 for final eval
        # Evaluate
        if args.eval_freq > 0 and ((global_step - args.num_envs) // args.eval_freq) < (global_step // args.eval_freq):
            eval_d = evaluate(args, eval_envs, get_eval_action, logger, eval_output_dir,
                              max_episode_steps, global_step, pbar)
            if args.evaluate:
                break
            if args.save_model:
                ckpt = {
                    'encoder': encoder.state_dict(),
                    'actor': actor.state_dict(),
                    'critic': critic_target.state_dict(),
                    'log_alpha': log_alpha,
                    'global_step': global_step,
                }
                torch.save(ckpt, model_path)
                print(f"Step {global_step}: model checkpoint saved to {model_path}")
                # Eval success varies checkpoint to checkpoint, so the final weights are
                # not necessarily the best ones seen -- keep the best separately.
                score = (float(eval_d['eval/success_at_end']), float(eval_d['eval/return']))
                if score > best_eval:
                    best_eval = score
                    torch.save(ckpt, best_model_path)
                    print(f"Step {global_step}: new best (success_at_end={score[0]:.2f}, "
                          f"return={score[1]:.2f}) saved to {best_model_path}")

        # Collect. Before learning_starts, a fresh run explores with random actions; a
        # warm-started run (--checkpoint) collects with the loaded policy instead, so the
        # buffer holds its (good) trajectories rather than junk by the time updates begin.
        if global_step < args.learning_starts and args.checkpoint is None:
            actions = envs.action_space.sample()
        else:
            actions = get_rollout_action(obs['rgb'], obs['state'])

        if isinstance(actions, torch.Tensor):
            actions = actions.to(sim_device)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = {'rgb': next_obs['rgb'].clone(), 'state': next_obs['state'].clone()}
        if 'priv' in next_obs:
            real_next_obs['priv'] = next_obs['priv'].clone()

        # Determine bootstrap behavior
        if args.bootstrap_at_done == 'never':
            need_final_obs = terminations | truncations
            dones = terminations | truncations
        elif args.bootstrap_at_done == 'always':
            need_final_obs = terminations | truncations
            dones = torch.zeros_like(terminations, dtype=torch.bool)
        else: # 'on_truncation' - only stop bootstrap on true termination, bootstrap on truncation
            need_final_obs = truncations & (~terminations)
            dones = terminations

        if "final_info" in infos:
            real_next_obs['rgb'][need_final_obs] = infos["final_observation"]['rgb'][need_final_obs]
            real_next_obs['state'][need_final_obs] = infos["final_observation"]['state'][need_final_obs]
            if 'priv' in real_next_obs:
                real_next_obs['priv'][need_final_obs] = infos["final_observation"]['priv'][need_final_obs]

        transition = TensorDict(
            observations=obs,
            next_observations=real_next_obs,
            actions=torch.as_tensor(actions, device=device, dtype=torch.float),
            rewards=torch.as_tensor(rewards, device=device, dtype=torch.float),
            dones=dones,
            batch_size=rewards.shape[0],
            device=device,
        )
        rb.extend(transition)

        # Setting next as current obs
        obs = next_obs

        # Training updates
        if global_step > args.learning_starts:
            for grad_step in range(args.num_updates):
                data = rb.sample(args.batch_size)

                # update critic and encoder and actor entropy
                out_main = update_main(data)
                encoded_rgb = out_main.pop("encoded_rgb", None)

                # update actor (policy)
                if grad_step % args.policy_frequency == 0:
                    out_main.update(update_actor(data, encoded_rgb))

                # update target networks
                if grad_step % args.target_network_frequency == 0:
                    with torch.no_grad():
                        torch._foreach_lerp_(critic_target_params, critic_online_params, args.tau)

                d.update(out_main)

        # Log
        if "final_info" in infos:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                d[f"train/{k}"] = v[done_mask].float().mean()
            # logging for terminal bar
            max_ep_ret = max(infos["final_info"]["episode"]["return"][done_mask])
            avg_returns.extend(infos["final_info"]["episode"]["return"][done_mask])
            desc = f"global_step={global_step}, episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})"
            # Calculate wall_time metrics
            sps = global_step / logger.wall_time
            d["time/sps"] = sps
            pbar.set_description(f"{sps: 4.4f} sps, " + desc)
            logger.log(d=d, step=global_step)

        # Increment counters
        pbar.update(args.num_envs)
        global_step += args.num_envs

    # Upload final checkpoint to wandb
    if args.save_model:
        if os.path.exists(model_path):
            model_name = f"model_{args.agent_name}_{args.env_id}_{args.seed}"
            logger.upload_checkpoint(model_path=model_path, model_name=model_name)
        else:
            print(f"WARNING: Checkpoint file not found at {model_path}, skipping upload")

    print("Finishing logger...")
    logger.close()
    print("Starting cleanup...")
    try:
        envs.close()
        eval_envs.close()
    except:
        pass
    print("Cleanup complete. Exiting.")
