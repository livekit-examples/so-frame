"""Weights & Biases logging and the eval loop.

Both vendored from Squint (https://github.com/aalmuzairee/squint), unchanged apart from
dropping the privileged-state plumbing.
"""

import glob
import os
import time
from collections import defaultdict

import torch
import wandb


class Logger:
    def __init__(self, log_wandb=False):
        self.log_wandb = log_wandb
        self.start_time = time.perf_counter()
        self.total_eval_time = 0   # to subtract from total wall_time

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


def evaluate(args, eval_envs, get_action_fn, logger, eval_output_dir, max_episode_steps,
             global_step, pbar):
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
