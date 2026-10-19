"""Calculate corpus-level WER and CER from transcription directories"""

import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder import CTCTextEncoder


def read_transcription(path):
    """
    Read and normalize one transcription file
    """
    text = path.read_text(encoding="utf-8").strip()
    return CTCTextEncoder.normalize_text(text)


def calculate_metrics(ground_truth_dir, predictions_dir):
    """
    Calculate corpus-level WER and CER for matching transcription files
    """
    ground_truth_dir = Path(ground_truth_dir)
    predictions_dir = Path(predictions_dir)

    if not ground_truth_dir.is_dir():
        raise FileNotFoundError(
            f"Ground-truth directory does not exist: {ground_truth_dir}"
        )

    if not predictions_dir.is_dir():
        raise FileNotFoundError(
            f"Predictions directory does not exist: {predictions_dir}"
        )

    reference_files = sorted(ground_truth_dir.glob("*.txt"))

    if not reference_files:
        raise RuntimeError(
            f"No transcription files found in {ground_truth_dir}"
        )

    total_word_errors = 0
    total_words = 0

    total_char_errors = 0
    total_chars = 0

    for reference_path in reference_files:
        prediction_path = predictions_dir / reference_path.name

        if not prediction_path.is_file():
            raise FileNotFoundError(
                f"Prediction is missing for: {reference_path.name}"
            )

        reference = read_transcription(reference_path)
        prediction = read_transcription(prediction_path)

        wer_result = calc_wer(reference, prediction)
        cer_result = calc_cer(reference, prediction)

        total_word_errors += wer_result["dist"]
        total_words += wer_result["length"]

        total_char_errors += cer_result["dist"]
        total_chars += cer_result["length"]

    if total_words == 0:
        raise RuntimeError("Ground-truth transcriptions contain no words.")

    if total_chars == 0:
        raise RuntimeError("Ground-truth transcriptions contain no characters.")

    wer = total_word_errors / total_words
    cer = total_char_errors / total_chars

    return wer, cer


def main():
    parser = argparse.ArgumentParser(description="Calculate WER and CER for ASR predictions")

    parser.add_argument("--ground-truth", required=True, help="Directory containing ground-truth .txt files")

    parser.add_argument("--predictions", required=True, help="Directory containing predicted .txt files")

    args = parser.parse_args()

    wer, cer = calculate_metrics(ground_truth_dir=args.ground_truth, predictions_dir=args.predictions)

    print(f"WER: {wer:.6f} ({wer * 100:.2f}%)")
    print(f"CER: {cer:.6f} ({cer * 100:.2f}%)")


if __name__ == "__main__":
    main()