import argparse

import torch

from megatron.bridge.recipes.qwen_vl.qwen3_vl import qwen3_vl_8b_pretrain_config
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.vlm_step import forward_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # model parallelism
    parser.add_argument("--tensor-model-parallel-size", type=int, default=4)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-parallel", action="store_true", default=False)

    # training
    parser.add_argument("--train-iters", type=int, default=20)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)

    # learning rate
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--lr-warmup-iters", type=int, default=10)
    parser.add_argument("--lr-decay-iters", type=int, default=None)

    # dataset
    parser.add_argument("--dataset-type", type=str, default="hf",
                        choices=["mock", "hf", "preloaded", "energon"])

    # freeze
    parser.add_argument("--freeze-language-model", action="store_true", default=True)
    parser.add_argument("--no-freeze-language-model", dest="freeze_language_model", action="store_false")
    parser.add_argument("--freeze-vision-model", action="store_true", default=True)
    parser.add_argument("--no-freeze-vision-model", dest="freeze_vision_model", action="store_false")
    parser.add_argument("--freeze-vision-projection", action="store_true", default=False)
    parser.add_argument("--no-freeze-vision-projection", dest="freeze_vision_projection", action="store_false")

    # logging
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--log-throughput", action="store_true", default=False)
    parser.add_argument("--log-throughput-to-tensorboard", action="store_true", default=False)
    parser.add_argument("--throughput-window-size", type=int, default=100)
    parser.add_argument("--log-timers-to-tensorboard", action="store_true", default=False)
    parser.add_argument("--log-memory-to-tensorboard", action="store_true", default=False)
    parser.add_argument("--log-params-norm", action="store_true", default=False)
    parser.add_argument("--log-progress", action="store_true", default=False)
    parser.add_argument("--timing-log-level", type=int, default=0, choices=[0, 1, 2])

    # wandb
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-exp-name", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)

    # checkpoint
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-dir", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = qwen3_vl_8b_pretrain_config(
        dataset_type=args.dataset_type,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=args.sequence_parallel,
        train_iters=args.train_iters,
        seq_length=args.seq_length,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_iters=args.lr_warmup_iters,
        lr_decay_iters=args.lr_decay_iters if args.lr_decay_iters else args.train_iters,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        freeze_language_model=args.freeze_language_model,
        freeze_vision_model=args.freeze_vision_model,
        freeze_vision_projection=args.freeze_vision_projection,
    )
    cfg.model.share_embeddings_and_output_weights = False

    # logging
    cfg.logger.log_interval = args.log_interval
    cfg.logger.log_throughput = args.log_throughput
    cfg.logger.log_throughput_to_tensorboard = args.log_throughput_to_tensorboard
    cfg.logger.throughput_window_size = args.throughput_window_size
    cfg.logger.log_timers_to_tensorboard = args.log_timers_to_tensorboard
    cfg.logger.log_memory_to_tensorboard = args.log_memory_to_tensorboard
    cfg.logger.log_params_norm = args.log_params_norm
    cfg.logger.log_progress = args.log_progress
    cfg.logger.timing_log_level = args.timing_log_level

    # wandb
    if args.wandb_project:
        cfg.logger.wandb_project = args.wandb_project
    if args.wandb_exp_name:
        cfg.logger.wandb_exp_name = args.wandb_exp_name
    if args.wandb_entity:
        cfg.logger.wandb_entity = args.wandb_entity

    # checkpoint
    if args.checkpoint_dir:
        cfg.checkpoint.save = args.checkpoint_dir
        cfg.checkpoint.load = args.checkpoint_dir

    pretrain(config=cfg, forward_step_func=forward_step)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
