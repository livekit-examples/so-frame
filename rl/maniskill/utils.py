"""Training helper wrappers, vendored from Squint's `utils.py`
(https://github.com/aalmuzairee/squint). Robot/task-agnostic: resolution downsampling
and color jitter on RGB observations (upstream's buffer-memory helper was dropped)."""

import gymnasium as gym
import torch
import torch.nn.functional as F

import torchvision

# ---------------------------  Wrappers --------------------------------------#

class SplitPrivilegedStateWrapper(gym.ObservationWrapper):
    """Split FlattenRGBDObservationWrapper's merged 'state' vector into the actor-safe
    proprio part (`state`, the first n_proprio dims: the env's agent obs) and the
    privileged remainder (`priv`: everything from `_get_obs_extra` -- ground-truth
    item/bin/tcp poses, randomized physics params). For asymmetric actor-critic: the
    critic trains on state+priv, while the actor (and any deployed policy) sees only
    `state`, so nothing privileged leaks into what must run on the real robot.

    Apply directly after FlattenRGBDObservationWrapper(state=True), whose 'state' is
    the flattened obs dict in insertion order: agent first, extra second -- that
    ordering is what makes the fixed split index valid.
    """

    def __init__(self, env, n_proprio: int):
        self.base_env = env.unwrapped
        super().__init__(env)
        self.n_proprio = n_proprio
        assert self.base_env._init_raw_obs["state"].shape[-1] > n_proprio, (
            "state vector has nothing beyond proprio -- was the env created with a "
            "'+state' obs_mode so _get_obs_extra is populated?"
        )
        self.base_env.update_obs_space(self.observation(dict(self.base_env._init_raw_obs)))

    def observation(self, observation: dict):
        state = observation["state"]
        observation = dict(observation)
        observation["state"] = state[..., : self.n_proprio]
        observation["priv"] = state[..., self.n_proprio :]
        return observation


class DownsampleObsWrapper(gym.ObservationWrapper):
    """Downsamples RGB observations from render_size to target_size using area interpolation.

    Expects input in (B, H, W, C) format.
    """
    def __init__(self, env, target_size):
        super().__init__(env)
        self.target_size = target_size
        # Update observation space
        old_rgb_space = self.observation_space['rgb']
        C = old_rgb_space.shape[-1]
        self.observation_space['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(target_size, target_size, C), dtype=old_rgb_space.dtype
        )

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) or (H, W, C)
        if rgb.shape[-2] == self.target_size:
            return obs  # Already at target size

        # Handle batched and unbatched cases
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W) for interpolate
        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb.float(), size=(self.target_size, self.target_size), mode='area').to(torch.uint8)
        # (B, C, H, W) -> (B, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs



class ColorJitterWrapper(gym.ObservationWrapper):
    """Applies random color jitter to RGB observations for sim2real robustness.

    Expects input in (B, H, W, C) format. C may be a multiple of 3 (multiple cameras
    concatenated along the channel axis by `FlattenRGBDObservationWrapper`, e.g. 6 for a
    wrist + overhead camera pair): `torchvision.transforms.ColorJitter`'s hue/saturation
    ops hard-require exactly 3 channels, so each camera's 3 channels are jittered as its
    own independent image rather than jittering the stack as one 6+-channel blob.
    """
    def __init__(self, env, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        super().__init__(env)
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def observation(self, obs):
        rgb = obs['rgb']  # (B, H, W, C) or (H, W, C) uint8, C a multiple of 3

        # Handle batched and unbatched cases
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        # (B, H, W, C) -> (B, C, H, W) for ColorJitter
        rgb = rgb.permute(0, 3, 1, 2)
        num_channels = rgb.shape[1]
        assert num_channels % 3 == 0, f"expected a multiple of 3 channels (one or more RGB cameras), got {num_channels}"
        rgb = torch.cat(
            [self.jitter(rgb[:, c:c + 3].float() / 255.0) for c in range(0, num_channels, 3)],
            dim=1,
        )
        # (B, C, H, W) -> (B, H, W, C)
        rgb = rgb.permute(0, 2, 3, 1)

        # Back to uint8
        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


class SensorAugWrapper(gym.ObservationWrapper):
    """Camera sensor-realism augmentation, for sim2real. Clean sim renders lack three
    things the real USB cameras have: h264 compression artifacts (blocking/softening),
    sensor noise, and auto-exposure/white-balance drift. This adds a blocking proxy,
    Gaussian noise, gamma, and per-channel gain so the policy stops depending on clean
    pixels. Complements ColorJitterWrapper (hue/sat/brightness); apply this AFTER it.

    Input (B, H, W, C) uint8, C a multiple of 3 -- each camera's 3 channels are augmented
    independently (like ColorJitterWrapper). gamma/white-balance/noise are per-image
    (each env its own); blur/blocking are per-batch (one draw/step) for speed."""

    def __init__(self, env, gamma=(0.7, 1.4), wb=0.1, noise_std=0.04,
                 blur_prob=0.3, blur_sigma=(0.3, 0.9), block_prob=0.3):
        super().__init__(env)
        self.gamma_range = gamma
        self.wb = wb
        self.noise_std = noise_std
        self.blur_prob = blur_prob
        self.blur_sigma = blur_sigma
        self.block_prob = block_prob

    @staticmethod
    def _gauss_kernel(sigma, device, dtype):
        x = torch.arange(-2, 3, device=device, dtype=dtype)
        k1 = torch.exp(-(x ** 2) / (2 * sigma * sigma))
        k1 = k1 / k1.sum()
        return torch.outer(k1, k1)  # (5, 5)

    def observation(self, obs):
        rgb = obs['rgb']
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0  # (B, C, H, W)
        B, C, H, W = rgb.shape
        dev, dt = rgb.device, rgb.dtype
        assert C % 3 == 0, f"expected a multiple of 3 channels, got {C}"

        out = []
        for c in range(0, C, 3):
            img = rgb[:, c:c + 3]                                       # (B, 3, H, W)
            gamma = torch.empty(B, 1, 1, 1, device=dev, dtype=dt).uniform_(*self.gamma_range)
            img = img.clamp_min(1e-6).pow(gamma)                        # exposure/gamma
            gain = torch.empty(B, 3, 1, 1, device=dev, dtype=dt).uniform_(1 - self.wb, 1 + self.wb)
            img = img * gain                                           # per-channel white balance
            if torch.rand(1, device=dev).item() < self.blur_prob:
                sigma = float(torch.empty(1, device=dev).uniform_(*self.blur_sigma).item())
                k = self._gauss_kernel(sigma, dev, dt).expand(3, 1, 5, 5)
                img = F.conv2d(img, k, padding=2, groups=3)            # soft blur
            if torch.rand(1, device=dev).item() < self.block_prob:
                img = F.interpolate(img, size=(max(1, H // 2), max(1, W // 2)), mode='nearest')
                img = F.interpolate(img, size=(H, W), mode='nearest')  # blocking / compression proxy
            std = torch.empty(B, 1, 1, 1, device=dev, dtype=dt).uniform_(0, self.noise_std)
            img = img + torch.randn_like(img) * std                    # sensor noise
            out.append(img.clamp(0, 1))

        rgb = (torch.cat(out, dim=1) * 255).to(torch.uint8).permute(0, 2, 3, 1)
        if squeeze:
            rgb = rgb.squeeze(0)
        obs['rgb'] = rgb
        return obs


# --- DINOv2 feature caching --------------------------------------------------#
_DINO_BACKBONE = {}


def load_dino_backbone(device):
    """Frozen DINOv2 ViT-S/14 (registers), loaded once per device and shared."""
    key = str(device)
    if key not in _DINO_BACKBONE:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        m = m.to(device).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        _DINO_BACKBONE[key] = m
    return _DINO_BACKBONE[key]


class DinoTokenWrapper(gym.ObservationWrapper):
    """Replace the RGB image obs with FROZEN DINOv2 patch tokens (feature caching).

    Runs the ViT once per env-step HERE, at the env boundary, so the replay buffer stores
    tokens and the trainable policy consumes cached tokens, the ViT never runs inside the
    gradient update (that's the whole speed win, and it lets the update stay compilable).

    Input obs['rgb']: (B, H, W, C) uint8, C = 3*num_cams (H a multiple of 14).
    Output obs['rgb']: (B, n_tok, EMBED) bf16 tokens, n_tok = num_cams*(H/14)^2, EMBED=384.
    Each camera's 3 channels run through DINOv2 forward_features separately; the per-camera
    token grids are concatenated. Apply this LAST in the obs pipeline (after downsample /
    jitter / sensor-aug, which need images). The trainable DinoHead consumes these tokens.
    """
    PATCH = 14
    EMBED = 384

    def __init__(self, env, device=None):
        super().__init__(env)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dino = load_dino_backbone(self.device)
        shp = self.observation_space['rgb'].shape  # (H, W, C)
        H, C = shp[-2], shp[-1]
        assert C % 3 == 0, f"expected 3*num_cams channels, got {C}"
        self.num_cams = C // 3
        self.dino_res = max(2 * self.PATCH, (H // self.PATCH) * self.PATCH)
        self.grid = self.dino_res // self.PATCH
        self.n_tok = self.num_cams * self.grid * self.grid
        self.register_buffer_tensors()
        self.observation_space['rgb'] = gym.spaces.Box(
            low=-3.4e38, high=3.4e38, shape=(self.n_tok, self.EMBED), dtype="float32"
        )

    def register_buffer_tensors(self):
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def observation(self, obs):
        rgb = obs['rgb']
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)
        x = rgb.permute(0, 3, 1, 2).float() / 255.0  # (B, C, H, W)
        mean, std = self._mean.to(x.device), self._std.to(x.device)
        toks = []
        amp = torch.autocast("cuda", dtype=torch.bfloat16) if x.is_cuda else torch.autocast("cpu", dtype=torch.bfloat16)
        with amp:
            for c in range(self.num_cams):
                cam = x[:, 3 * c: 3 * c + 3]
                if cam.shape[-1] != self.dino_res:
                    cam = F.interpolate(cam, size=(self.dino_res, self.dino_res),
                                        mode="bilinear", align_corners=False)
                cam = (cam - mean) / std
                toks.append(self._dino.forward_features(cam)["x_norm_patchtokens"])
        tokens = torch.cat(toks, dim=1).to(torch.bfloat16)  # (B, n_tok, EMBED)
        if squeeze:
            tokens = tokens.squeeze(0)
        obs['rgb'] = tokens
        return obs
