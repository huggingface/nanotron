import argparse
import os
import subprocess
import sys

import lovely_tensors as lt

lt.monkey_patch()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process the commands for training or evaluation.")
    parser.add_argument("command", choices=["train", "eval", "debug-train", "debug-eval"], help="Command to execute")
    parser.add_argument("--config-file", help="Path to the configuration file")
    # parser.add_argument("--output-dir", help="Path to the output directory", default="/fsx/ferdinandmom/ferdinand-hf/nanotron/examples/fsdp/experiments/local")
    parser.add_argument("--no-wandb", action="store_true", help="Disable WANDB logging")
    parser.add_argument(
        "--eval_ckpt",
        nargs="?",
        default=None,
        help="Checkpoint path to evaluate (optional, required for eval and debug-eval)",
    )
    parser.add_argument("--nproc_per_node", default=1, help="Nb GPU nodes")
    parser.add_argument("--port", default=1234, help="Master Port")
    parser.add_argument("--master_port", default=29601, help="Master Port")
    parser.add_argument("--dp", type=int, default=1)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--tp", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    if args.command in ["eval", "debug-eval"] and not args.eval_ckpt:
        print("Please provide a checkpoint path to evaluate")
        sys.exit(1)

    use_wandb = 0 if args.no_wandb else 1

    env_vars = os.environ.copy()
    env_vars["FI_PROVIDER"] = "efa"
    env_vars["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    env_vars["USE_WANDB"] = str(use_wandb)

    # if args.command in ["train", "debug-train"]:
    #     config = get_config_from_file(args.config_file)
    #     out_dir_path = f"{args.output_dir}/{config.experiment_logger.wandb_logger.wandb_project}"
    #     # mamba/configs/test_2180333.yaml => test_2180333
    #     config_name = os.path.basename(args.config_file)
    #     config_name = os.path.splitext(config_name)[0]
    #     output_log_dir = f"{out_dir_path}/{config_name}"

    #     directories = [
    #         out_dir_path,
    #         output_log_dir,
    #         f"{output_log_dir}/checkpoints",
    #         f"{output_log_dir}/configs",
    #         f"{output_log_dir}/logs",
    #         f"{output_log_dir}/lighteval",
    #         f"{output_log_dir}/lighteval/s3_tmp",
    #         f"{output_log_dir}/lighteval/slurm_scripts",
    #     ]

    #     for dir_path in directories:
    #         if not os.path.exists(dir_path):
    #             os.makedirs(dir_path)

    if args.command == "debug-train":
        command = (
            f"debugpy-run "
            f"-p {args.port} -m torch.distributed.run "
            f"-- --nproc_per_node={args.nproc_per_node} --master_port={args.master_port} ../././run_train.py "
            f"--config-file={args.config_file}"
        )
    elif args.command == "debug-eval":
        command = (
            f"debugpy-run "
            f"-p {args.port} -m torch.distributed.run "
            f"-- --nproc_per_node={args.nproc_per_node} --master_port={args.master_port} mamba/run_generate.py "
            f"--pp {args.pp} --tp {args.tp} --ckpt-path {args.eval_ckpt}"
        )
    elif args.command == "eval":
        command = (
            f"torchrun --nproc_per_node={args.nproc_per_node} --master_port={args.master_port} "
            f"mamba/run_generate.py --pp {args.pp} --tp {args.tp} --ckpt-path {args.eval_ckpt}"
        )
    elif args.command == "train":
        command = (
            f"torchrun --nproc_per_node={args.nproc_per_node} --master_port={args.master_port} "
            f"../../run_train.py --config-file={args.config_file}"
        )

    subprocess.run(command, shell=True, env=env_vars)


if __name__ == "__main__":
    main()
