import argparse
import json
import logging
from pathlib import Path

from src.logger import CometMLWriter


def flatten_config(config, prefix=""):
    """
    Convert nested config into Comet-friendly scalar parameters.
    """
    result = {}

    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            result.update(flatten_config(value, full_key))

        elif isinstance(value, list):
            # Avoid passing lists/dicts directly to Comet parameters.
            result[full_key] = json.dumps(value, ensure_ascii=False)

        else:
            result[full_key] = value

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Log saved inference results to CometML."
    )

    parser.add_argument(
        "results_path",
        type=str,
        help="Path to JSON produced by inference.py",
    )

    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Comet experiment name",
    )

    parser.add_argument(
        "--project-name",
        type=str,
        default="asr-decoder-tests",
        help="Comet project name",
    )

    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Comet workspace",
    )

    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="Comet logging mode",
    )

    args = parser.parse_args()

    results_path = Path(args.results_path)

    if not results_path.is_file():
        raise FileNotFoundError(
            f"Results file not found: {results_path}"
        )

    with results_path.open("r", encoding="utf-8") as file:
        results = json.load(file)

    if "metrics" not in results:
        raise KeyError(
            f"{results_path} does not contain 'metrics'"
        )

    config = results.get("config", {})

    run_name = (
        args.run_name
        if args.run_name is not None
        else results_path.stem
    )

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # CometMLWriter expects trainer.resume_from when deciding
    # whether this is a resumed experiment.
    #
    # We also flatten the saved inference config so that Comet
    # receives only scalar/string parameters.
    project_config = {
        "trainer": {
            "resume_from": None,
        },
        **flatten_config(config),
    }

    writer = CometMLWriter(
        logger=logger,
        project_config=project_config,
        project_name=args.project_name,
        workspace=args.workspace,
        run_name=run_name,
        mode=args.mode,
        log_code=False,
        log_graph=False,
        auto_metric_logging=False,
        auto_param_logging=False,
    )

    for part, part_metrics in results["metrics"].items():
        writer.set_step(
            step=0,
            mode=part,
        )

        writer.add_scalars(
            {
                metric_name: float(value)
                for metric_name, value in part_metrics.items()
            }
        )

    # Important for OfflineExperiment:
    # finish writing and create the archive.
    if hasattr(writer, "exp"):
        writer.exp.end()

    print()
    print(f"Logged inference results: {results_path}")
    print(f"Run name: {run_name}")
    print(f"Project: {args.project_name}")
    print(f"Mode: {args.mode}")


if __name__ == "__main__":
    main()