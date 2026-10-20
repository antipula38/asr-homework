# ASR Homework

DeepSpeech2 + CTC speech recognition with three decoding modes:

- greedy CTC (`argmax`);
- hand-written CTC beam search (`beam`);
- beam search with a 3-gram KenLM language model (`beam_lm`).

The final inference checkpoint is stored with Git LFS and the language model is downloaded by `download_resources.py`.

## Installation

Python 3.11 is used for the project. Dependencies are locked in `uv.lock`.

Install Git LFS first.

Ubuntu / Google Colab:

```bash
sudo apt-get update
sudo apt-get install -y git-lfs
git lfs install
```

macOS:

```bash
brew install git-lfs
git lfs install
```

Clone the repository and create the environment:

```bash
git clone https://github.com/antipula38/asr-homework.git
cd asr-homework

pip install uv
uv sync --locked --python 3.11
```

No additional manual dependency installation is required. In particular, the compatible `pyarrow` version is already pinned in `pyproject.toml` / `uv.lock`.

## Download inference resources

Run:

```bash
uv run --no-sync python download_resources.py
```

The script prepares:

```text
saved/final_model/model_best.pth
resources/lm/3-gram.pruned.1e-7.arpa
```

The checkpoint is pulled through Git LFS. The 3-gram LibriSpeech KenLM model is downloaded and extracted automatically.

## Custom dataset format

`CustomDirDataset` expects:

```text
dataset/
├── audio/
│   ├── UtteranceID1.wav
│   ├── UtteranceID2.flac
│   └── UtteranceID3.mp3
└── transcriptions/          # optional
    ├── UtteranceID1.txt
    ├── UtteranceID2.txt
    └── UtteranceID3.txt
```

`audio/` is required. `transcriptions/` may be absent when only predictions are needed.

Prediction file names preserve the utterance ids.

## Final-model inference

The final decoder is beam search with KenLM:

```bash
uv run --no-sync python inference.py \
  inferencer.from_pretrained=saved/final_model/model_best.pth \
  datasets.test.data_dir=data/custom_dataset \
  inferencer.save_path=outputs/custom_dataset \
  dataloader.batch_size=4 \
  dataloader.num_workers=0
```

Predictions are written to:

```text
outputs/custom_dataset/test/
```

For example:

```text
data/custom_dataset/audio/example_1.flac
→ outputs/custom_dataset/test/example_1.txt
```

### Decoder overrides

The same checkpoint can be evaluated with all implemented decoding modes:

```bash
# Greedy CTC
uv run --no-sync python inference.py \
  inferencer.from_pretrained=saved/final_model/model_best.pth \
  datasets.test.data_dir=data/custom_dataset \
  inferencer.save_path=outputs/argmax \
  inferencer.decoder=argmax \
  dataloader.num_workers=0 \
  lm.enabled=false
```

```bash
# Beam search, without LM
uv run --no-sync python inference.py \
  inferencer.from_pretrained=saved/final_model/model_best.pth \
  datasets.test.data_dir=data/custom_dataset \
  inferencer.save_path=outputs/beam \
  inferencer.decoder=beam \
  inferencer.beam_size=10 \
  dataloader.num_workers=0 \
  lm.enabled=false
```

For a quick smoke test of `beam_lm`, a smaller `inferencer.beam_size` such as `5` or `10` can be used. The final command above uses beam size `200`.

## WER and CER

If `transcriptions/` exists, calculate metrics with:

```bash
uv run --no-sync python calc_metrics.py \
  --ground-truth data/custom_dataset/transcriptions \
  --predictions outputs/custom_dataset/test
```

The script reports WER and CER using the saved prediction text files.

## Training

A short training sanity check can be run with:

```bash
uv run --no-sync python train.py \
  model=deepspeech2 \
  metrics=train_asr \
  datasets=onebatchtest \
  trainer.n_epochs=1 \
  trainer.epoch_len=2 \
  trainer.device=cpu \
  dataloader.batch_size=2 \
  dataloader.num_workers=0 \
  writer=cometml \
  writer.mode=offline \
  writer.run_name=smoke_train \
  trainer.override=true
```

`trainer.device=cpu` is intentional for this portable smoke test: the CTC loss used by this project is not available natively on Apple MPS in the tested environment.

Regular experiments can use the Hydra configs in `src/configs/` and override model, dataset, optimizer, scheduler, augmentation, and trainer parameters from the command line.


## Demo

Open `demo.ipynb` in a **fresh Google Colab runtime**.

To evaluate another dataset, change only:

```python
DATASET_URL = "YOUR_DATASET_URL"
```

`DATASET_URL` must point to a `.zip`, `.tar`, `.tar.gz`, or `.tgz` archive, or to a public Google Drive file containing such an archive.

Then select:

```text
Runtime → Run all
```

The notebook automatically:

1. installs Git LFS;
2. clones the repository;
3. installs the locked Python 3.11 environment;
4. downloads the checkpoint and KenLM model;
5. runs `inference.py` on a sample dataset;
6. runs `calc_metrics.py` on the sample;
7. downloads and extracts the dataset from `DATASET_URL`;
8. runs final `beam_lm` inference;
9. calculates WER/CER when `transcriptions/` is present.

No changes other than `DATASET_URL` are required for the custom-dataset evaluation.
