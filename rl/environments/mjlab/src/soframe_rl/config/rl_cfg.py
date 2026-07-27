"""PPO (rsl-rl) runner config, built from ``rl/train.toml``."""

from __future__ import annotations

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from soframe_rl.train_params import PARAMS


def so101_pick_place_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  runner = PARAMS["runner"]
  policy = PARAMS["policy"]
  hidden = tuple(policy["hidden_dims"])

  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=hidden,
      activation=policy["activation"],
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": policy["init_std"],
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=hidden,
      activation=policy["activation"],
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(**PARAMS["algorithm"]),
    experiment_name=runner["experiment_name"],
    save_interval=runner["save_interval"],
    num_steps_per_env=runner["num_steps_per_env"],
    max_iterations=runner["max_iterations"],
    # TensorBoard by default (no external account/API key needed); logs + checkpoints
    # go under logs/rsl_rl/<experiment_name>/<timestamp>/. Override with --agent.logger.
    logger="tensorboard",
  )
