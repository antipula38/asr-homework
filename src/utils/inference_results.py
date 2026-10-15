import json
from pathlib import Path

from omegaconf import OmegaConf


def save_inference_results(logs, config, output_path):
    """
    Save inference metrics and resolved run config to JSON.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = {
        part: {
            key: float(value)
            for key, value in part_logs.items()
        }
        for part, part_logs in logs.items()
    }

    result = {
        "metrics": metrics,
        "config": OmegaConf.to_container(
            config,
            resolve=True,
        ),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )