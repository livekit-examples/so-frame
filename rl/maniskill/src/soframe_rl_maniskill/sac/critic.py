"""Distributional (C51) Q-ensemble critic.

Vendored unchanged from Squint (https://github.com/aalmuzairee/squint), which uses
``tensordict.from_modules`` + ``torch.vmap`` so the ensemble is one batched forward rather than
``num_q`` separate ones.

Training-side only, deliberately. The critic never runs on the robot, so it stays out of the
shared ``soframe_nets`` package -- what deploy needs is only the encoder and actor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules

from soframe_nets.actor import Projection, weight_init


class Critic(nn.Module):
    """Distributional C51 Ensemble-Q-network critic with vmap optimizations."""

    def __init__(self, n_obs, n_state, n_act, num_atoms, v_min, v_max, num_q=2, device=None,
                 rgb_dim=50):
        super().__init__()
        self.num_atoms = num_atoms
        self.num_q = num_q
        self.v_min = v_min
        self.v_max = v_max
        self.q_support = torch.linspace(v_min, v_max, num_atoms, device=device)

        # Same projection the actor uses, at the same width, so both read the encoder's
        # features through an identically shaped fusion.
        self.proj = Projection(n_obs, n_state, device=device, rgb_dim=rgb_dim)
        self.proj.apply(weight_init)

        q_input_dim = self.proj.repr_dim + n_act

        q_nets = [self._build_q_network(q_input_dim, num_atoms, device=device) for _ in range(num_q)]
        for qn in q_nets:
            qn.apply(weight_init)

        self.q_params = from_modules(*q_nets, as_module=True)  # stacked params for optimizer + vmap
        # meta template for vmap dispatch; kept out of parameters()/state_dict()
        object.__setattr__(self, '_q_meta', self._build_q_network(q_input_dim, num_atoms, device="meta"))
        object.__setattr__(self, '_q_repr', repr(q_nets[0]))

    def __repr__(self):
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
        """Expected Q-values [num_q, batch]; detach_critic freezes critic weights but keeps grad through actions (actor PG)."""
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
        """C51 categorical projection [num_q, batch, num_atoms]; called under no_grad for targets."""
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

        logits = self.forward(rgb_features, state, actions)  # [num_q, batch, atoms]
        next_dists = F.softmax(logits, dim=-1)

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
