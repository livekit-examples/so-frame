"""The SAC training loop.

Vendored from Squint (https://github.com/aalmuzairee/squint): autotuned entropy, distributional C51
critic ensemble, torch.compile + CUDA graphs, high update-to-data ratio. The algorithm is untouched;
the encoder/actor come from ``soframe_policy`` so the deployed policy is the trained one.

Local additions: warm-start (--checkpoint) with optional --reset_alpha, best-checkpoint tracking,
and an optional encoder LR group + grad clipping (which the transformer head needs and the CNN does
not).
"""

import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import wandb
from tensordict import TensorDict, from_module
from tensordict.nn import CudaGraphModule
from torchrl.data import LazyTensorStorage, ReplayBuffer

from soframe_policy import Actor
from soframe_policy import checkpoint as ckpt_io
from soframe_policy.encoders import ENCODERS

from .build import make_envs
from .critic import Critic
from .logger import Logger, evaluate
from .replay import ObsOnlyReplay


def train(args):
    args = args.resolve()

    # MUST precede any env/agent build: the controller reads config.ARM_SPEED_SCALE when the agent's
    # controller configs are constructed.
    from .. import config
    config.ARM_SPEED_SCALE = args.arm_speed_scale

    # Same: pick_place reads config.COLOR_CUBE / COLOR_BIN when it builds the scene.
    if args.object_colors == "distinct":
        config.COLOR_CUBE = config.BLUE
        config.COLOR_BIN = config.YELLOW
        print("[config] --object_colors distinct: cube blue, bin yellow "
              "(hue cue the res-32 squint CNN needs; costs sim2real fidelity)")

    run_name = args.exp_name or f"{args.env_id}__{args.encoder}__{args.seed}__{int(time.time())}"
    model_path = os.path.abspath(f"runs/{run_name}/ckpt.pt")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    best_model_path = model_path.replace("ckpt.pt", "ckpt_best.pt")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    bundle = make_envs(args, run_name, device)
    envs, eval_envs = bundle.envs, bundle.eval_envs
    max_episode_steps = bundle.max_episode_steps

    if not args.evaluate:
        print("Running training")
        if args.track:
            # NOT named `config`: that would shadow the config module imported above.
            wandb_config = vars(args)
            wandb_config["env_cfg"] = dict(**bundle.env_kwargs, num_envs=args.num_envs,
                                           env_id=args.env_id, reward_mode="normalized_dense",
                                           env_horizon=max_episode_steps,
                                           partial_reset=args.partial_reset)
            wandb_config["eval_env_cfg"] = dict(**bundle.eval_env_kwargs,
                                                num_envs=args.num_eval_envs,
                                                env_id=args.env_id,
                                                reward_mode="normalized_dense",
                                                env_horizon=max_episode_steps,
                                                partial_reset=args.eval_partial_reset)
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                       sync_tensorboard=False, config=wandb_config, name=run_name, save_code=True,
                       group=args.wandb_group,
                       tags=[args.wandb_group, args.encoder, args.env_id, f"seed={args.seed}"])
    else:
        print("Running evaluation")
    logger = Logger(log_wandb=(args.track and not args.evaluate))

    # -- Networks ------------------------------------------------------------------------
    action_space = envs.unwrapped.single_action_space
    encoder_cls = ENCODERS[args.encoder]

    # Constructor arguments beyond (num_cams, res). MUST be recorded in the checkpoint: dino_global's
    # pool choice changes no tensor shape, so it cannot be inferred back from the weights.
    encoder_kwargs = {"pool": args.dino_pool} if args.encoder == "dino_global" else {}

    def new_encoder(dev=device):
        return encoder_cls(bundle.num_cams, res=args.res, device=dev, **encoder_kwargs)

    def new_actor(dev=device):
        return Actor(encoder.repr_dim, bundle.n_state, bundle.n_act,
                     action_space.low, action_space.high, device=dev,
                     rgb_dim=encoder_cls.RGB_PROJ_DIM)

    encoder = new_encoder()
    actor = new_actor()
    critic = Critic(n_obs=encoder.repr_dim, n_state=bundle.n_state, n_act=bundle.n_act,
                    num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
                    num_q=args.num_q, device=device, rgb_dim=encoder_cls.RGB_PROJ_DIM)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.detach().exp()
        alpha_optimizer = optim.Adam([log_alpha], lr=args.alpha_lr,
                                     capturable=args.cudagraphs and not args.compile)
    else:
        log_alpha = None
        alpha = torch.as_tensor(args.alpha, device=device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        actor.load_state_dict(ckpt['actor'])
        if 'critic' in ckpt:
            critic.load_state_dict(ckpt['critic'])
        if 'log_alpha' in ckpt and log_alpha is not None and not args.reset_alpha:
            with torch.no_grad():
                log_alpha.copy_(ckpt['log_alpha'])
                alpha.copy_(log_alpha.exp())
        print(f"Loaded checkpoint from {args.checkpoint} at step {ckpt.get('global_step')}"
              + (" (log_alpha reset to default for fresh exploration)" if args.reset_alpha else ""))

    # Inference copies weight-share with the trained modules via from_module.
    encoder_detach = new_encoder()
    encoder_eval = new_encoder().eval()
    from_module(encoder).data.to_module(encoder_detach)
    from_module(encoder).data.to_module(encoder_eval)

    actor_detach = new_actor()
    actor_eval = new_actor().eval()
    from_module(actor).data.to_module(actor_detach)
    from_module(actor).data.to_module(actor_eval)

    critic_target = Critic(n_obs=encoder.repr_dim, n_state=bundle.n_state, n_act=bundle.n_act,
                           num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
                           num_q=args.num_q, device=device, rgb_dim=encoder_cls.RGB_PROJ_DIM)
    critic_target.load_state_dict(critic.state_dict())
    critic_online_params = list(critic.parameters())
    critic_target_params = list(critic_target.parameters())

    def get_rollout_action(rgb, state):
        # obs may be on cpu under raytraced (cpu backend) while networks live on `device`
        rgb_feat = encoder_detach(rgb.to(device))
        action, _, _ = actor_detach.get_action(rgb_feat, state.to(device))
        return action

    def get_eval_action(rgb, state):
        rgb_feat = encoder_eval(rgb.to(device))
        return actor_eval.get_eval_action(rgb_feat, state.to(device))

    # The encoder trains with the critic (the actor sees it detached), optionally at its own LR.
    critic_optimizer = optim.Adam(
        [
            {"params": list(critic.parameters()), "lr": args.q_lr},
            {"params": list(encoder.parameters()), "lr": args.encoder_lr},
        ],
        lr=args.q_lr, capturable=args.cudagraphs and not args.compile,
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr,
                                 capturable=args.cudagraphs and not args.compile)

    if args.obs_only_replay:
        rb = ObsOnlyReplay(args.buffer_size, args.num_envs,
                           device=device,
                           storage_device="cpu" if args.replay_storage == "cpu" else device,
                           prefetch=args.replay_prefetch)
        if rb.capacity != args.buffer_size:
            print(f"[replay] buffer_size {args.buffer_size} -> {rb.capacity} "
                  f"(rounded down to {rb.iters} whole iterations of {args.num_envs} envs)")
        print(f"[replay] {rb.capacity:,} transitions = {args.replay_episodes:g} episodes/env "
              f"on {args.replay_storage}"
              + (f", prefetch {args.replay_prefetch}" if args.replay_storage == "cpu" else ""))
    else:
        rb = ReplayBuffer(storage=LazyTensorStorage(args.buffer_size, device=device))

    print("-----------------------")
    print(args)
    print("-----------------------")
    print(f"Encoder: {args.encoder} ({encoder_cls.__name__}), res {args.res}, "
          f"repr_dim {encoder.repr_dim}, rgb_proj {encoder_cls.RGB_PROJ_DIM}")
    print("-----------------------")
    for mod in [encoder, actor, critic]:
        print(mod)
    print(f"Task: {args.env_id}, Control mode: {envs.unwrapped._control_mode}")
    print(f"Cameras: {bundle.num_cams}, Actions: {bundle.n_act}")
    print(f"Proprio: {bundle.proprio.describe()}")
    print(f"Device: {device}")
    print("-----------------------")

    def save(path, global_step):
        """Weights plus the metadata deploy needs to rebuild this exact policy."""
        ckpt_io.save(
            path, encoder, actor,
            num_cams=bundle.num_cams, res=args.res, n_act=bundle.n_act,
            action_low=action_space.low, action_high=action_space.high,
            proprio=bundle.proprio, global_step=global_step,
            encoder_kwargs=encoder_kwargs,
            # Training-only extras; deploy ignores them, a resumed run uses them.
            extra={"critic": critic_target.state_dict(),
                   "log_alpha": log_alpha,
                   "encoder_name": args.encoder},
        )

    # -- Update steps --------------------------------------------------------------------
    # bf16 autocast on whichever device we are on, so the loop stays smoke-testable on a CPU box.
    amp_device = "cuda" if device.type == "cuda" else "cpu"

    def update_main(data):
        with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
            with torch.no_grad():
                next_obs = encoder(data["next_observations"]['rgb'])
                next_state = data["next_observations"]['state']
                next_state_actions, next_state_log_pi, _ = actor.get_action(next_obs, next_state)

                bootstrap = (~data["dones"]).float()
                discount = args.gamma
                rewards = data["rewards"].flatten()

                entropy_bonus = alpha * next_state_log_pi.flatten()
                rewards_with_entropy = rewards - bootstrap.flatten() * discount * entropy_bonus

                target_distributions = critic_target.categorical(
                    next_obs, next_state, next_state_actions,
                    rewards_with_entropy, bootstrap, discount
                )

            obs = encoder(data["observations"]['rgb'])
            state = data["observations"]['state']

            q_outputs = critic(obs, state, data["actions"])
            q_log_probs = F.log_softmax(q_outputs, dim=-1)
            q_losses = -torch.sum(target_distributions * q_log_probs, dim=-1).mean(dim=-1)
            critic_loss = q_losses.sum()

            with torch.no_grad():
                q_probs = F.softmax(q_outputs, dim=-1)
                q_values = torch.sum(q_probs * critic.q_support, dim=-1)
                q_max = q_values.max()
                q_min = q_values.min()

        critic_optimizer.zero_grad()
        critic_loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                list(critic.parameters()) + list(encoder.parameters()), args.grad_clip)
        critic_optimizer.step()

        if args.autotune:
            with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
                with torch.no_grad():
                    _, log_pi, _ = actor.get_action(obs, state)
                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

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
        with torch.amp.autocast(amp_device, dtype=torch.bfloat16):
            state = data["observations"]["state"]
            obs = encoded_rgb

            pi, log_pi, _ = actor.get_action(obs, state)
            q_values = critic.get_q_values(obs, state, pi, detach_critic=True)

            # Mean (No CDQ)
            critic_value = q_values.mean(dim=0)

            actor_loss = (alpha * log_pi - critic_value).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        if args.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()), args.grad_clip)
        actor_optimizer.step()

        return TensorDict(actor_loss=actor_loss.detach())

    _update_main, _update_actor = update_main, update_actor
    _get_rollout_action, _get_eval_action = get_rollout_action, get_eval_action
    if args.compile:
        _update_main = torch.compile(update_main)
        _update_actor = torch.compile(update_actor)
        _get_rollout_action = torch.compile(get_rollout_action)
        _get_eval_action = torch.compile(get_eval_action)

    if args.cudagraphs:
        _update_main = CudaGraphModule(_update_main)
        _update_actor = CudaGraphModule(_update_actor)

    # -- Rollout / learn loop ------------------------------------------------------------
    obs, _ = envs.reset(seed=args.seed)
    eval_envs.reset(seed=args.seed)

    global_step = 0
    # cpu backend under raytraced training
    sim_device = torch.device("cpu") if args.visual_fidelity == "raytraced" else device
    pbar = tqdm.tqdm(total=args.total_timesteps, desc="steps")
    max_ep_ret = -float("inf")
    avg_returns = deque(maxlen=20)
    desc = ""
    d = {}
    best_eval = (-float("inf"), -float("inf"))   # (success_at_end, return)

    for iteration in range(args.num_total_iterations + 2):  # +2 for final eval
        if args.eval_freq > 0 and ((global_step - args.num_envs) // args.eval_freq) < (global_step // args.eval_freq):
            eval_d = evaluate(args, eval_envs, _get_eval_action, logger, bundle.eval_output_dir,
                              max_episode_steps, global_step, pbar)
            if args.evaluate:
                break
            if args.save_model:
                save(model_path, global_step)
                print(f"Step {global_step}: model checkpoint saved to {model_path}")
                # keep the best checkpoint separately; final weights aren't always the best
                score = (float(eval_d['eval/success_at_end']), float(eval_d['eval/return']))
                if score > best_eval:
                    best_eval = score
                    save(best_model_path, global_step)
                    print(f"Step {global_step}: new best (success_at_end={score[0]:.2f}, "
                          f"return={score[1]:.2f}) saved to {best_model_path}")

        # fresh run explores randomly before learning_starts; a warm-started run collects with the loaded policy
        if global_step < args.learning_starts and args.checkpoint is None:
            actions = envs.action_space.sample()
            # At num_envs=1 the sampled action comes back unbatched, (n_act,) rather than
            # (1, n_act), and the replay TensorDict then rejects it on a batch-dim mismatch.
            if getattr(actions, "ndim", 2) == 1:
                actions = actions[None]
        else:
            actions = _get_rollout_action(obs['rgb'], obs['state'])

        if isinstance(actions, torch.Tensor):
            actions = actions.to(sim_device)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        real_next_obs = {'rgb': next_obs['rgb'].clone(), 'state': next_obs['state'].clone()}

        if args.bootstrap_at_done == 'never':
            need_final_obs = terminations | truncations
            dones = terminations | truncations
        elif args.bootstrap_at_done == 'always':
            need_final_obs = terminations | truncations
            dones = torch.zeros_like(terminations, dtype=torch.bool)
        else:  # 'on_truncation' - only stop bootstrap on true termination, bootstrap on truncation
            need_final_obs = truncations & (~terminations)
            dones = terminations

        if "final_info" in infos:
            real_next_obs['rgb'][need_final_obs] = infos["final_observation"]['rgb'][need_final_obs]
            real_next_obs['state'][need_final_obs] = infos["final_observation"]['state'][need_final_obs]

        act_t = torch.as_tensor(actions, device=device, dtype=torch.float)
        rew_t = torch.as_tensor(rewards, device=device, dtype=torch.float)
        if args.obs_only_replay:
            # next_obs is not stored: it is the same env's observation one iteration later.
            # `need_final_obs` is exactly the mask where that identity breaks (the real next
            # observation is the pre-reset final_observation), so those slots get dropped.
            rb.extend(obs['rgb'], obs['state'], act_t, rew_t, dones, need_final_obs)
        else:
            rb.extend(TensorDict(
                observations=obs,
                next_observations=real_next_obs,
                actions=act_t,
                rewards=rew_t,
                dones=dones,
                batch_size=rewards.shape[0],
                device=device,
            ))

        obs = next_obs

        if global_step > args.learning_starts:
            for grad_step in range(args.num_updates):
                data = rb.sample(args.batch_size)

                out_main = _update_main(data)
                encoded_rgb = out_main.pop("encoded_rgb", None)

                if grad_step % args.policy_frequency == 0:
                    out_main.update(_update_actor(data, encoded_rgb))

                if grad_step % args.target_network_frequency == 0:
                    with torch.no_grad():
                        torch._foreach_lerp_(critic_target_params, critic_online_params, args.tau)

                d.update(out_main)

        # Episode metrics only exist on a truncation boundary, but the SAC losses in `d` refresh
        # every iteration, so the two cadences stay separate.
        episode_ended = "final_info" in infos
        if episode_ended:
            final_info = infos["final_info"]
            done_mask = infos["_final_info"]
            for k, v in final_info["episode"].items():
                d[f"train/{k}"] = v[done_mask].float().mean()
            max_ep_ret = max(infos["final_info"]["episode"]["return"][done_mask])
            avg_returns.extend(infos["final_info"]["episode"]["return"][done_mask])
            desc = (f"global_step={global_step}, "
                    f"episodic_return={torch.tensor(avg_returns).mean(): 4.2f} (max={max_ep_ret: 4.2f})")

        if d and (episode_ended or iteration % args.log_freq == 0):
            sps = global_step / logger.wall_time
            d["time/sps"] = sps
            pbar.set_description(f"{sps: 4.4f} sps, " + desc)
            logger.log(d=d, step=global_step)
            # Cleared so the next point carries fresh losses and the episode metrics stay a sparse
            # series rather than a stale value repeated until the next truncation.
            d = {}

        pbar.update(args.num_envs)
        global_step += args.num_envs

    if args.save_model and os.path.exists(model_path):
        logger.upload_checkpoint(
            model_path=model_path,
            model_name=f"model_{args.encoder}_{args.env_id}_{args.seed}",
        )

    print("Finishing logger...")
    logger.close()
    print("Starting cleanup...")
    try:
        if hasattr(rb, "close"):
            rb.close()      # stop the replay prefetch thread before tearing the sim down
        envs.close()
        eval_envs.close()
    except Exception:
        pass
    print("Cleanup complete. Exiting.")
