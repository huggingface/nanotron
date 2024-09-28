# flake8: noqa: C901
import argparse

from lighteval.main_nanotron import main

from nanotron.config import Config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-config-path",
        type=str,
        required=True,
        help="Path to the Nanotron checkpoint YAML or python config file, potentially on S3",
    )
    parser.add_argument(
        "--lighteval-config-path",
        type=str,
        required=True,
        help="Path to an optional YAML or python Lighteval config",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args, unknowns = parser.parse_known_args()
    main(
        checkpoint_config_path=args.checkpoint_config_path,
        lighteval_config_path=args.lighteval_config_path,
        cache_dir=args.cache_dir,
    )
