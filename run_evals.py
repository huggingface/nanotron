import argparse

from lighteval.main_nanotron import nanotron


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-config-path",
        type=str,
        required=True,
        help="Path to the brr checkpoint YAML or python config file, potentially on S3",
    )
    parser.add_argument(
        "--lighteval-override",
        type=str,
        default=None,
        help="Path to an optional YAML or python Lighteval config to override part of the checkpoint Lighteval config",
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
    nanotron(
        checkpoint_config_path=args.checkpoint_config_path,
        lighteval_config_path=args.lighteval_override,
        cache_dir=args.cache_dir,
    )
