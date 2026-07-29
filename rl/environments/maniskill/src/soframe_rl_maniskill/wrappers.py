"""Observation wrappers for the training obs pipeline.

Adapted from Squint (https://github.com/aalmuzairee/squint): resolution downsampling and colour
jitter, plus this repo's camera sensor-realism augmentation and DINOv2 tokenisation.

These run once per env step, outside the gradient update, so the expensive parts (the frozen ViT
especially) are paid per collected transition rather than per minibatch, and what lands in the
replay buffer is already encoder-ready.

Transforms with a deploy counterpart in ``soframe_policy``, which must not drift on resize mode,
normalisation or camera ordering (the tokeniser is imported, not reimplemented):

    DownsampleObsWrapper  <->  SquintEncoder.preprocess     (area resize)
    DinoTokenWrapper      <->  DinoPatchEncoder.preprocess  (both call soframe_policy tokenize)

The jitter/augmentation wrappers have no deploy counterpart: the real cameras supply their own
noise.
"""

import gymnasium as gym
import torch
import torch.nn.functional as F
import torchvision

from soframe_policy.encoders import dino_global, dino_patch


class DownsampleObsWrapper(gym.ObservationWrapper):
    """Downsample RGB observations to target_size (area interpolation); expects (B, H, W, C)."""

    def __init__(self, env, target_size):
        super().__init__(env)
        self.target_size = target_size
        old_rgb_space = self.observation_space['rgb']
        C = old_rgb_space.shape[-1]
        self.observation_space['rgb'] = gym.spaces.Box(
            low=0, high=255, shape=(target_size, target_size, C), dtype=old_rgb_space.dtype
        )

    def observation(self, obs):
        rgb = obs['rgb']
        if rgb.shape[-2] == self.target_size:
            return obs

        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        rgb = rgb.permute(0, 3, 1, 2)
        rgb = F.interpolate(rgb.float(), size=(self.target_size, self.target_size), mode='area').to(torch.uint8)
        rgb = rgb.permute(0, 2, 3, 1)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


class ColorJitterWrapper(gym.ObservationWrapper):
    """Random colour jitter on RGB for sim2real. Input (B, H, W, C); C a multiple of 3
    (cameras concatenated on the channel axis). ColorJitter needs exactly 3 channels, so each
    camera's 3 channels are jittered independently."""

    def __init__(self, env, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05):
        super().__init__(env)
        self.jitter = torchvision.transforms.ColorJitter(brightness, contrast, saturation, hue)

    def observation(self, obs):
        rgb = obs['rgb']

        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)

        rgb = rgb.permute(0, 3, 1, 2)
        num_channels = rgb.shape[1]
        assert num_channels % 3 == 0, \
            f"expected a multiple of 3 channels (one or more RGB cameras), got {num_channels}"
        rgb = torch.cat(
            [self.jitter(rgb[:, c:c + 3].float() / 255.0) for c in range(0, num_channels, 3)],
            dim=1,
        )
        rgb = rgb.permute(0, 2, 3, 1)

        rgb = (rgb.clamp(0, 1) * 255).to(torch.uint8)

        if squeeze:
            rgb = rgb.squeeze(0)

        obs['rgb'] = rgb
        return obs


class SensorAugWrapper(gym.ObservationWrapper):
    """Camera sensor-realism augmentation for sim2real (blocking proxy, Gaussian noise,
    gamma, per-channel gain). Apply AFTER ColorJitterWrapper. Input (B, H, W, C) uint8,
    C a multiple of 3; each camera augmented independently."""

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
        return torch.outer(k1, k1)

    def observation(self, obs):
        rgb = obs['rgb']
        squeeze = rgb.dim() == 3
        if squeeze:
            rgb = rgb.unsqueeze(0)
        rgb = rgb.permute(0, 3, 1, 2).float() / 255.0
        B, C, H, W = rgb.shape
        dev, dt = rgb.device, rgb.dtype
        assert C % 3 == 0, f"expected a multiple of 3 channels, got {C}"

        out = []
        for c in range(0, C, 3):
            img = rgb[:, c:c + 3]
            gamma = torch.empty(B, 1, 1, 1, device=dev, dtype=dt).uniform_(*self.gamma_range)
            img = img.clamp_min(1e-6).pow(gamma)
            gain = torch.empty(B, 3, 1, 1, device=dev, dtype=dt).uniform_(1 - self.wb, 1 + self.wb)
            img = img * gain
            if torch.rand(1, device=dev).item() < self.blur_prob:
                sigma = float(torch.empty(1, device=dev).uniform_(*self.blur_sigma).item())
                k = self._gauss_kernel(sigma, dev, dt).expand(3, 1, 5, 5)
                img = F.conv2d(img, k, padding=2, groups=3)
            if torch.rand(1, device=dev).item() < self.block_prob:
                img = F.interpolate(img, size=(max(1, H // 2), max(1, W // 2)), mode='nearest')
                img = F.interpolate(img, size=(H, W), mode='nearest')  # blocking/compression proxy
            std = torch.empty(B, 1, 1, 1, device=dev, dtype=dt).uniform_(0, self.noise_std)
            img = img + torch.randn_like(img) * std
            out.append(img.clamp(0, 1))

        rgb = (torch.cat(out, dim=1) * 255).to(torch.uint8).permute(0, 2, 3, 1)
        if squeeze:
            rgb = rgb.squeeze(0)
        obs['rgb'] = rgb
        return obs


class DinoTokenWrapper(gym.ObservationWrapper):
    """Replace RGB observations with frozen DINOv2 patch tokens, cached.

    The ViT runs here, once per env step, so the backbone never runs inside a gradient update.
    Only the head trains.

    Input (B, H, W, 3*num_cams) uint8; output (B, n_tok, 384) bf16. Apply LAST in the pipeline:
    anything above needs images, and this emits features. Tokenisation is
    ``soframe_policy.encoders.dino_patch.tokenize``, the call the deployed policy makes.
    """

    def __init__(self, env, res, device=None):
        super().__init__(env)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.res = res
        shp = self.observation_space['rgb'].shape
        C = shp[-1]
        assert C % 3 == 0, f"expected 3*num_cams channels, got {C}"
        self.num_cams = C // 3
        self.n_tok = dino_patch.token_count(self.num_cams, res)
        # Load now, not on the first observation: a missing torch.hub cache must fail at startup
        # rather than mid-rollout.
        dino_patch.load_backbone(self.device)
        self.observation_space['rgb'] = gym.spaces.Box(
            low=-3.4e38, high=3.4e38, shape=(self.n_tok, dino_patch.EMBED), dtype="float32"
        )

    def observation(self, obs):
        rgb = obs['rgb']
        squeeze = rgb.dim() == 3
        tokens = dino_patch.tokenize(rgb, self.res, num_cams=self.num_cams, device=self.device)
        if squeeze:
            tokens = tokens.squeeze(0)
        obs['rgb'] = tokens
        return obs


class DinoGlobalWrapper(gym.ObservationWrapper):
    """Replace RGB observations with frozen DINOv2 global vectors, cached.

    The collapsed counterpart of DinoTokenWrapper: one vector per camera (two under
    pool="cls_mean") instead of a patch grid. Same placement rule, LAST in the pipeline.

    Input (B, H, W, 3*num_cams) uint8; output (B, n_vec, 384) bf16, i.e. 1.5 KB per observation
    against the patch grid's 216 KB.
    """

    def __init__(self, env, res, device=None, pool="cls"):
        super().__init__(env)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.res = res
        self.pool = pool
        shp = self.observation_space['rgb'].shape
        C = shp[-1]
        assert C % 3 == 0, f"expected 3*num_cams channels, got {C}"
        self.num_cams = C // 3
        self.n_vec = dino_global.vector_count(self.num_cams, pool)
        # Load now, not on the first observation: a missing torch.hub cache must fail at startup
        # rather than mid-rollout.
        dino_global.load_backbone(self.device)
        self.observation_space['rgb'] = gym.spaces.Box(
            low=-3.4e38, high=3.4e38, shape=(self.n_vec, dino_global.EMBED), dtype="float32"
        )

    def observation(self, obs):
        rgb = obs['rgb']
        squeeze = rgb.dim() == 3
        vecs = dino_global.embed_global(rgb, self.res, num_cams=self.num_cams,
                                        device=self.device, pool=self.pool)
        if squeeze:
            vecs = vecs.squeeze(0)
        obs['rgb'] = vecs
        return obs
