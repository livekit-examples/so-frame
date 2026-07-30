"""Training configuration, one dataclass for every encoder.

``--encoder`` and the resolution / buffer-size defaults that follow from it are the only structural
differences between architectures; ``resolve()`` derives them.
"""

from dataclasses import dataclass
from typing import Optional

from soframe_policy.encoders import ENCODERS

# Per-encoder defaults. `res` is the squinted image size for the CNN and the per-camera DINOv2
# input resolution for the patch head.
#
# `num_updates` is a tuned value rather than a structural consequence of the architecture, so keep
# it paired with the `batch_size` default of 512 and 1024 `num_envs`: it is a ratio against 1024
# envs, and changing num_envs without scaling it silently changes the update-to-data ratio.
#
# `buffer_size` is deliberately absent: it is derived from --replay_episodes for every encoder, so
# retention is equal and a comparison between them measures the architecture.
ENCODER_DEFAULTS = {
    "squint":      dict(res=32,  render_size=128, num_updates=64),
    "dino_patch":  dict(res=168, render_size=168, num_updates=32),
    # The dense-vs-collapsed control for dino_patch: same backbone, resolution and num_updates, so
    # the only difference is whether the patch grid survives. num_updates MUST match dino_patch's
    # or an update-ratio difference confounds what the run measures.
    "dino_global": dict(res=168, render_size=168, num_updates=32),
}


@dataclass
class Args:
    exp_name: Optional[str] = "baseline"
    """the name of this experiment"""
    encoder: str = "squint"
    """vision encoder: 'squint' (CNN over a squinted image stack) or 'dino_patch' (self-attention
    over frozen DINOv2 patch tokens). Sets the res/buffer_size/num_updates defaults above and is
    recorded in the checkpoint, so deploy is not told which architecture it is."""
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
    """on warm-start, reset entropy temperature to default so fine-tuning keeps exploring; --no-reset_alpha inherits the checkpoint's"""

    # Environment specific arguments
    env_id: str = "SOFramePickPlaceBin-v1"
    """the id of the environment"""
    env_domain_randomization: bool = True
    """adds domain randomization flag if env supports it"""
    randomize_colors: bool = False
    """also randomize bar/bin colors per scene build (on top of --object_colors); pair with --reconfiguration_freq"""
    object_colors: str = "black"
    """cube/bin colour scheme. "black" matches the real rig (both config.BLACK, ~4-5% reflectance
    matte plastic). "distinct" paints the cube blue and the bin yellow, which the squint CNN needs:
    at res 32 the cube is one pixel, so hue is the only cue it has. Costs sim2real fidelity, so it
    is not the default; dino_patch resolves the cube as a shape at res 112/168 and stays on
    "black"."""
    overhead_camera_fov: Optional[float] = None
    """override the overhead camera's base FOV, in DEGREES (measured via rl/calibrate)"""
    wrist_camera_fov: Optional[float] = None
    """override the wrist camera's base FOV, in DEGREES; same sourcing as overhead's"""
    overhead_camera_pos_offset: Optional[tuple[float, float, float]] = None
    """position correction for the overhead camera, meters in the camera-local frame (+X = view direction)"""
    overhead_camera_rot_offset: Optional[tuple[float, float, float]] = None
    """rotation correction for the overhead camera, roll/pitch/yaw in DEGREES (camera-local)"""
    visual_fidelity: str = "raster"
    """rendering fidelity (see envs/base_random_env.py): "raster" realistic PBR + soft shadow, "flat" fast shadowless, "raytraced" eval-only on the cpu backend"""
    num_envs: int = 1024
    """the number of parallel environments"""
    num_eval_envs: int = 35
    """the number of parallel evaluation environments. 35 rather than a round 32 so the eval
    video tiles without gaps: RecordEpisode uses nrows = int(sqrt(num_envs)), so 35 is exactly
    5 rows x 7 columns, where 32 leaves a ragged last column."""
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
    log_freq: int = 10
    """how often, in iterations, to push the SAC losses to wandb. Separate from episode metrics,
    which only exist at a truncation boundary (every env_horizon iterations, since partial_reset is
    off): tying losses to that boundary logs one point per env_horizon iterations."""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    control_mode: Optional[str] = None
    """the control mode to use for the environment"""
    obs_mode: Optional[str] = "rgb"
    """the observation output mode of the environment. Plain "rgb": the segmentation buffer only
    feeds the greenscreen overlay, which is off (see RandomizationConfig.apply_overlay). Pass
    --obs_mode rgb+segmentation if you turn the overlay back on."""
    sim_backend: str = "gpu"
    """physics/render backend. 'gpu' for real training (needs CUDA); 'cpu' supports a single
    env and exists so the loop can be smoke-tested on a machine without a GPU. Forced to 'cpu'
    by --visual_fidelity raytraced."""
    arm_speed_scale: float = 1.0
    """multiplier on arm/rail delta limits (gripper unscaled). 1.0 = the measured real servo
    speed (0.05 rad/step arm, 0.007 m/step rail at 10 Hz). For arm-speed ablations only."""
    render_mode: Optional[str] = "all"
    """the rendering mode of the environment, could be rgb or all"""
    render_size: Optional[int] = None
    """square size to render from env (HxW), before downsampling. Defaults per --encoder."""
    res: Optional[int] = None
    """encoder input resolution: the squinted image size for 'squint' (16/32/64), or the
    per-camera DINOv2 resolution for 'dino_patch' (a multiple of 14). Defaults per --encoder."""
    apply_jitter: bool = True
    """applies color jitter to all input RGB observations (better for sim2real)"""
    sensor_aug: bool = True
    """camera sensor-realism augmentation (compression/noise/gamma) on top of color jitter"""

    # Algorithm specific arguments
    total_timesteps: int = 12_000_000
    """total timesteps of the experiments. A from-scratch squint CNN does not land its first
    success until ~7M, so shorter budgets finish before the task is learned."""
    dino_pool: str = "cls"
    """--encoder dino_global only: what "one vector" means. "cls" is the CLS token, "mean" the mean
    over patch tokens, "cls_mean" both concatenated (two vectors per camera instead of one).
    Ignored by other encoders."""
    replay_episodes: float = 2.0
    """replay retention, in EPISODES per env. The buffer holds
    `replay_episodes * config.EPISODE_HORIZON * num_envs` transitions, the same formula for every
    encoder, so a comparison between them measures the architecture rather than the buffer."""
    buffer_size: Optional[int] = None
    """replay size in transitions. Leave unset: it is derived from --replay_episodes. Set it only to
    override that derivation."""
    replay_storage: str = "gpu"
    """where the replay lives: "gpu" (VRAM) or "cpu" (pinned host RAM, with a prefetch thread
    overlapping the gather and the PCIe copy with training).

    gpu by default: at dino_patch res 168, 512 envs, 2 episodes of a 200-step horizon, the
    single-copy buffer is 42 GiB and the run peaks at 59.5 GiB of a 97.9 GiB card, while host
    storage costs 24% throughput (267 vs 333 sps).

    Use cpu when the buffer will not fit: dino_patch at 1024 envs (84 GiB buffer, ~101 GiB total) or
    more than 2 episodes of retention. The prefetch holds the penalty near 24%."""
    replay_prefetch: int = 2
    """batches the replay keeps in flight when --replay_storage cpu. 0 makes sampling synchronous,
    the fallback if the prefetch thread is suspected of masking a bug."""
    obs_only_replay: bool = True
    """store each observation once and recover next_obs by an index offset, halving replay memory
    (see sac/replay.py; unit tests in tests/test_replay.py). Host storage is implemented only in
    this buffer. --no-obs_only_replay falls back to the torchrl buffer, which forces
    --replay_storage gpu."""
    batch_size: int = 512
    """the batch size of sample from the replay memory"""
    num_updates: Optional[int] = None
    """num updates per parallel env step. Defaults per --encoder."""
    learning_starts: int = 5_000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    alpha_lr: float = 3e-4
    """the learning rate of alpha for policy"""
    encoder_lr: Optional[float] = None
    """learning rate for the encoder, trained by the critic optimizer as its own param group.
    Defaults to q_lr; worth lowering (~1e-4) for the transformer head, whose representation must
    drift slowly or the critic's value estimates destabilize."""
    grad_clip: Optional[float] = None
    """max grad norm for the critic+encoder and actor updates. None disables. 10.0 is loose enough
    not to shrink normal transformer-head updates, tight enough to catch divergence."""
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
    """minimum value for distributional RL support. Must cover the discounted return range;
    the reward ladder tops out at REWARD_SUCCESS = 10."""
    v_max: float = 20.0
    """maximum value for distributional RL support"""

    # Optimizations
    compile: bool = True
    """whether to use torch.compile."""
    cudagraphs: bool = True
    """whether to use cudagraphs on top of compile. Forced off for 'dino_patch': the ViT's
    positional-embedding interpolation breaks static-shape capture."""

    # to be filled in runtime
    num_total_iterations: int = 0
    """the number of parallel envs steps given global total timesteps"""

    def resolve(self) -> "Args":
        """Fill in the per-encoder defaults and validate. Call once, before building anything."""
        if self.encoder not in ENCODERS:
            raise ValueError(f"--encoder must be one of {sorted(ENCODERS)}, got {self.encoder!r}")
        defaults = ENCODER_DEFAULTS[self.encoder]
        for field, value in defaults.items():
            if getattr(self, field) is None:
                setattr(self, field, value)
        if self.encoder_lr is None:
            self.encoder_lr = self.q_lr

        if self.object_colors not in ("black", "distinct"):
            raise ValueError(
                f"--object_colors must be black|distinct, got {self.object_colors!r}"
            )
        if self.encoder == "dino_global" and self.dino_pool not in ("cls", "mean", "cls_mean"):
            raise ValueError(
                f"--dino_pool must be cls|mean|cls_mean, got {self.dino_pool!r}"
            )
        if self.replay_storage not in ("cpu", "gpu"):
            raise ValueError(f"--replay_storage must be cpu|gpu, got {self.replay_storage!r}")
        if self.replay_storage == "cpu" and not self.obs_only_replay:
            raise ValueError(
                "--replay_storage cpu needs --obs_only_replay: host storage is implemented in "
                "sac/replay.py only, not in the torchrl buffer. Pass --replay_storage gpu to "
                "use the torchrl two-copy buffer, but note it cannot hold 2 episodes of "
                "dino_patch tokens in VRAM."
            )
        # Retention in episodes -> transitions; the buffer rounds to whole iterations itself.
        if self.buffer_size is None:
            from .. import config
            self.buffer_size = int(self.replay_episodes * config.EPISODE_HORIZON * self.num_envs)
        if self.sim_backend not in ("gpu", "cpu"):
            raise ValueError(f"--sim_backend must be gpu|cpu, got {self.sim_backend!r}")
        if self.sim_backend == "cpu" and (self.num_envs > 1 or self.num_eval_envs > 1):
            raise ValueError(
                "the cpu backend supports a single environment: pass --num_envs 1 "
                f"--num_eval_envs 1 (got {self.num_envs} / {self.num_eval_envs})"
            )
        if self.visual_fidelity not in ("flat", "raster", "raytraced"):
            raise ValueError(
                f"--visual_fidelity must be flat|raster|raytraced, got {self.visual_fidelity!r}"
            )
        if self.encoder == "squint" and self.res not in (16, 32, 64):
            raise ValueError(f"--res for the squint CNN must be 16, 32 or 64, got {self.res}")
        if self.encoder == "dino_patch":
            if self.res % 14 != 0:
                raise ValueError(
                    f"--res for dino_patch must be a multiple of DINOv2's 14px patch, got {self.res}"
                )
            if self.cudagraphs:
                # Not a silent override: the run would crash in graph capture otherwise.
                print("[args] --encoder dino_patch: forcing --no-cudagraphs (the ViT's "
                      "pos-embed interpolation breaks static-shape capture)")
                self.cudagraphs = False
        if self.render_size < self.res and self.encoder == "squint":
            raise ValueError(
                f"--render_size {self.render_size} is below --res {self.res}; the downsample "
                "wrapper only reduces resolution"
            )

        self.num_total_iterations = int(self.total_timesteps // self.num_envs)
        if self.num_updates <= 0:
            raise ValueError("--num_updates must be > 0 or no learning happens")
        return self
