"""Download resources required for ASR inference"""

import gzip
import shutil
import urllib.request
from pathlib import Path


LM_URL = (
    "https://www.openslr.org/resources/11/"
    "3-gram.pruned.1e-7.arpa.gz"
)

CHECKPOINT_URL = "saved/final_model/model_best.pth"

LM_PATH = Path("resources/lm/3-gram.pruned.1e-7.arpa")
CHECKPOINT_PATH = Path("saved/300h_640hs/model_best.pth")


def download_language_model():
    """
    Download and extract the LibriSpeech 3-gram language model
    """
    if LM_PATH.exists():
        print(f"Language model already exists: {LM_PATH}")
        return

    LM_PATH.parent.mkdir(parents=True, exist_ok=True)

    archive_path = LM_PATH.with_suffix(LM_PATH.suffix + ".gz")

    print(f"Downloading language model from {LM_URL}")
    urllib.request.urlretrieve(LM_URL, archive_path)

    print("Extracting language model...")

    with gzip.open(archive_path, "rb") as source:
        with LM_PATH.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    archive_path.unlink()

    print(f"Language model saved to: {LM_PATH}")


def download_checkpoint():
    """
    Download the final model checkpoint
    """
    if CHECKPOINT_PATH.exists():
        print(f"Checkpoint already exists: {CHECKPOINT_PATH}")
        return

    if not CHECKPOINT_URL:
        raise RuntimeError(
            "CHECKPOINT_URL is not configured in download_resources.py."
        )

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint from {CHECKPOINT_URL}")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)

    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")



def main():
    """
    Download all external resources required for inference
    """
    download_language_model()
    download_checkpoint()


if __name__ == "__main__":
    main()