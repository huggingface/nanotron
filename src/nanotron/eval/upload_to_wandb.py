import json
import s3fs
import wandb
import re
import argparse
from wandb.sdk.lib.runid import generate_id


def push_to_wandb(wandb_project, wandb_entity, model_name, results_path, train_step, consumed_tokens):
    s3 = s3fs.S3FileSystem(anon=False)
    all_metrics = {
        # basic X axis replacements for all metrics
        "consumed_tokens": consumed_tokens,
        "train_step": train_step,
    }

    for result_file in sorted(s3.ls(results_path)):
        if not result_file.endswith(".json"):
            continue

        with s3.open(result_file, "r") as f:
            results = json.loads(f.read())["results"]

            for benchmark, metrics in results.items():
                if benchmark == "all":
                    continue

                # extract dataset and config name
                match = re.search(r"\|(.*?)(?::(.*?))?\|", benchmark)
                if match:
                    dataset, subtask = match.groups()

                    for metric_name, metric_value in metrics.items():
                        if "_stderr" in metric_name:
                            continue
                        # wandb-friendly metric name
                        wandb_metric = f"{dataset}/{subtask}/{metric_name}" if subtask else f"{dataset}/{metric_name}"
                        all_metrics[wandb_metric] = metric_value

    run_id = f"{model_name}-{generate_id()}"

    # try to find the run in wandb and resume it
    api = wandb.Api()
    runs = api.runs(f"{wandb_entity}/{wandb_project}")
    for run in runs:
        if run.name == model_name:
            run_id = run.id
            break

    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=model_name,
        id=run_id,
        config={
            "model_name": model_name,
        },
        resume="allow",
    )

    # log all metrics for this checkpoint
    wandb.log(all_metrics)

    wandb.finish()

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Upload evaluation results to Weights & Biases.")
    parser.add_argument("--wandb_project", type=str, required=True, help="WandB project name.")
    parser.add_argument("--wandb_entity", type=str, required=True, help="WandB entity name.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model.")
    parser.add_argument("--results_path", type=str, required=True, help="S3 path to the results directory.")
    parser.add_argument("--train_step", type=int, required=True, help="Training step corresponding to the checkpoint.")
    parser.add_argument("--consumed_tokens", type=int, required=True, help="Total consumed tokens up to this checkpoint.")

    # Parse arguments
    args = parser.parse_args()

    # Call the main function with parsed arguments
    push_to_wandb(
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        model_name=args.model_name,
        results_path=args.results_path,
        train_step=args.train_step,
        consumed_tokens=args.consumed_tokens
    )
