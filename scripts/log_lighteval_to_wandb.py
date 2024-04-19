"""
This script use to log evaluation results to wandb.

python3 log_eval_results_to_wandb.py --eval-path /path/to/eval/results --wandb-project project_name --wandb-name run_name

The folder that contains the evaluation results should have the following structure:
- 5000:
    results_x.json # where x is the ligheval's evaluation number
- 10000:
    ...
...
"""
import argparse
import json
import os
from pathlib import Path

import wandb


def run(current_path: Path):
    def compute_avg_acc_of_a_benchmark(data, benchmark_prefix):
        sum_acc, sum_acc_norm, sum_acc_stderr, sum_acc_norm_stderr, count = 0, 0, 0, 0, 0
        for key, values in data.items():
            if f"{benchmark_prefix}:" in key:
                sum_acc += values["acc"]
                sum_acc_norm += values["acc_norm"]
                sum_acc_stderr += values["acc_stderr"]
                sum_acc_norm_stderr += values["acc_norm_stderr"]
                count += 1

        average_acc = sum_acc / count if count else 0
        return average_acc

    def compute_avg_acc_of_all_tasks(data):
        sum_acc, count = 0, 0
        for _, values in data.items():
            sum_acc += values["acc"]
            count += 1

        average_acc = sum_acc / count if count else 0
        return average_acc

    list_checkpoints = os.listdir(current_path)
    sorted_list_checkpoints = sorted(list_checkpoints, key=int)

    for item in sorted_list_checkpoints:
        item_path = os.path.join(current_path, item)
        if os.path.isdir(item_path):
            json_files = [f for f in os.listdir(item_path) if f.endswith(".json")]
            if len(json_files) == 1:
                json_file_path = os.path.join(item_path, json_files[0])

                with open(json_file_path, "r") as file:
                    eval_data = json.load(file)
                    iteration_step = eval_data["config_general"]["config"]["general"]["step"]
                    consumed_train_samples = eval_data["config_general"]["config"]["general"]["consumed_train_samples"]

                    logging_results = {}
                    for name, data in eval_data["results"].items():
                        logging_results[f"{name}_acc"] = data["acc"]

                    logging_results["mmlu:average_acc"] = compute_avg_acc_of_a_benchmark(eval_data["results"], "mmlu")
                    logging_results["arc:average_acc"] = compute_avg_acc_of_a_benchmark(eval_data["results"], "arc")
                    logging_results["all:average_acc"] = compute_avg_acc_of_all_tasks(eval_data["results"])

                    wandb.log(
                        {
                            **logging_results,
                            "iteration_step": iteration_step,
                            "consumed_train_samples": consumed_train_samples,
                        }
                    )

            elif len(json_files) > 1:
                print(f"More than one JSON file found in {item_path}. Skipping.")
            else:
                print(f"No JSON file found in {item_path}.")

        print(f"Checkpoint {item} is done. /n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", type=str, required=True, help="Path of the lighteval's evaluation results")
    parser.add_argument(
        "--wandb-project", type=str, help="Path of the lighteval's evaluation results", default="nanotron_evals"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        required=True,
        help="Path of the lighteval's evaluation results",
        default="sanity_evals",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    eval_path = args.eval_path
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name

    wandb.init(
        project=wandb_project,
        name=wandb_name,
        config={"eval_path": eval_path},
    )

    run(eval_path)
