"""Dataset implementation for user-provided ASR directories"""

from pathlib import Path
import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    """
    Load audio and optional transcriptions from a custom directory
    """

    SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3"}

    def __init__(self, data_dir, text_encoder=None, target_sr=16000, limit=None, max_audio_length=None, max_text_length=None, shuffle_index=False, instance_transforms=None):
        data_dir = Path(data_dir).expanduser().resolve()
        audio_dir = data_dir / "audio"
        transcription_dir = data_dir / "transcriptions"

        if not data_dir.exists():
            raise FileNotFoundError(
                f"Custom dataset directory does not exist: {data_dir}"
            )

        if not audio_dir.is_dir():
            raise FileNotFoundError(
                f"Expected audio directory: {audio_dir}"
            )

        audio_files = sorted(
            path
            for path in audio_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in self.SUPPORTED_EXTENSIONS
        )

        if not audio_files:
            raise RuntimeError(
                f"No .wav, .flac or .mp3 files found in {audio_dir}"
            )

        utterance_ids = [path.stem for path in audio_files]

        if len(utterance_ids) != len(set(utterance_ids)):
            raise RuntimeError(
                "Two audio files have the same UtteranceID "
            )

        index = []

        for audio_path in audio_files:
            utterance_id = audio_path.stem
            text_path = transcription_dir / f"{utterance_id}.txt"

            if text_path.is_file():
                text = text_path.read_text(encoding="utf-8").strip()
            else:
                text = ""

            audio_len = self._get_audio_length(audio_path)

            index.append(
                {
                    "path": str(audio_path),
                    "text": text,
                    "audio_len": audio_len,
                }
            )

        super().__init__(index=index, text_encoder=text_encoder, target_sr=target_sr, limit=limit, max_audio_length=max_audio_length, max_text_length=max_text_length, shuffle_index=shuffle_index, instance_transforms=instance_transforms)

    @staticmethod
    def _get_audio_length(audio_path):
        """
        Return duration in seconds
        """
        try:
            info = torchaudio.info(str(audio_path))

            if info.sample_rate > 0 and info.num_frames > 0:
                return info.num_frames / info.sample_rate

        except (AttributeError, RuntimeError):
            pass

        audio, sample_rate = torchaudio.load(str(audio_path))

        if sample_rate <= 0:
            raise RuntimeError(
                f"Invalid sample rate for {audio_path}"
            )

        return audio.shape[-1] / sample_rate