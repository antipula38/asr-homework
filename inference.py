import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, get_class

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH
from src.utils.inference_results import save_inference_results

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """Build the ASR inference pipeline and run prediction
    Args:
            config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup text_encoder
    text_encoder = instantiate(config.text_encoder)
    lm_config = config.get("lm")

    if lm_config is not None and lm_config.get("enabled", False):
        lm_path = Path(lm_config.path)

        if not lm_path.is_absolute():
            lm_path = ROOT_PATH / lm_path

        text_encoder.load_lm(
            lm_path=lm_path,
            alpha=lm_config.alpha,
            beta=lm_config.beta,
        )

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, text_encoder, device)

    # build model architecture, then print to console
    model = instantiate(config.model, n_tokens=len(text_encoder)).to(device)
    print(model)

    # get metrics
    metrics = {"inference": []}
    for metric_config in config.metrics.get("inference", []):
        # use text_encoder in metrics
        metrics["inference"].append(
            instantiate(metric_config, text_encoder=text_encoder)
        )

    # save_path for model predictions
    save_path = Path(config.inferencer.save_path)

    if not save_path.is_absolute():
        save_path = ROOT_PATH / save_path
    
    save_path.mkdir(exist_ok=True, parents=True)

    inferencer_cls = get_class(
        config.inferencer.get(
            "class_path",
            "src.trainer.Inferencer",
        )
    )
    inferencer = inferencer_cls(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        text_encoder=text_encoder,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=False,
    )

    logs = inferencer.run_inference()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"{full_key:15s}: {value}")
            
    results_path = config.inferencer.get("results_path")

    if results_path is not None:
        results_path = Path(results_path)

        if not results_path.is_absolute():
            results_path = ROOT_PATH / results_path

        save_inference_results(
            logs=logs,
            config=config,
            output_path=results_path,
        )

        print(f"Inference results saved to: {results_path}")


if __name__ == "__main__":
    main()
