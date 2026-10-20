#!/usr/bin/env python3
from __future__ import annotations

import gzip
import shutil
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

CHECKPOINT = ROOT / "saved/final_model/model_best.pth"

LM_DIR = ROOT / "resources/lm"
LM_PATH = LM_DIR / "3-gram.pruned.1e-7.arpa"
LM_GZ = LM_DIR / "3-gram.pruned.1e-7.arpa.gz"
LM_URL = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"


def run(command: list[str]) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 4096:
        return False

    text = path.read_text(encoding="utf-8", errors="ignore")
    return "version https://git-lfs.github.com/spec/v1" in text


def download_checkpoint() -> None:
    print("\n[1/2] Downloading final checkpoint with Git LFS...")

    if shutil.which("git-lfs") is None and shutil.which("git") is not None:
        # `git lfs` may still be available even if the executable is named differently,
        # so let the command below be the final check.
        pass

    try:
        run(["git", "lfs", "install"])
        run([
            "git",
            "lfs",
            "pull",
            "--include=saved/final_model/model_best.pth",
        ])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Git LFS is required to download model_best.pth. "
            "Install git-lfs and run this script again."
        ) from exc

    if not CHECKPOINT.exists():
        raise RuntimeError(
            f"Checkpoint was not downloaded: {CHECKPOINT.relative_to(ROOT)}"
        )

    if is_lfs_pointer(CHECKPOINT):
        raise RuntimeError(
            "model_best.pth is still a Git LFS pointer, not the real checkpoint."
        )

    print(
        f"Checkpoint ready: {CHECKPOINT.relative_to(ROOT)} "
        f"({CHECKPOINT.stat().st_size / 1024**2:.1f} MiB)"
    )


def download_lm() -> None:
    print("\n[2/2] Downloading LibriSpeech 3-gram KenLM...")

    LM_DIR.mkdir(parents=True, exist_ok=True)

    if LM_PATH.exists() and LM_PATH.stat().st_size > 1_000_000:
        print(
            f"LM already exists: {LM_PATH.relative_to(ROOT)} "
            f"({LM_PATH.stat().st_size / 1024**2:.1f} MiB)"
        )
        return

    print(f"Downloading: {LM_URL}")
    urllib.request.urlretrieve(LM_URL, LM_GZ)

    print("Extracting language model...")
    with gzip.open(LM_GZ, "rb") as source, LM_PATH.open("wb") as target:
        shutil.copyfileobj(source, target)

    LM_GZ.unlink(missing_ok=True)

    if not LM_PATH.exists() or LM_PATH.stat().st_size < 1_000_000:
        raise RuntimeError("Language model download/extraction failed.")

    print(
        f"LM ready: {LM_PATH.relative_to(ROOT)} "
        f"({LM_PATH.stat().st_size / 1024**2:.1f} MiB)"
    )


def main() -> None:
    download_checkpoint()
    download_lm()

    print("\nAll inference resources are ready.")


if __name__ == "__main__":
    main()
