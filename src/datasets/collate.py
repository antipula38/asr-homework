import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    audios = []
    spectrograms = []
    texts = []
    texts_encoded = []
    audio_paths = []
    audio_length = []
    spectrogram_length = []
    text_encoded_length = []

    for item in dataset_items:
        audios.append(item["audio"].permute(1, 0))
        spectrograms.append(item["spectrogram"].squeeze(0).permute(1, 0))
        texts.append(item["text"])
        texts_encoded.append(item["text_encoded"].permute(1, 0))
        audio_paths.append(item["audio_path"])
        audio_length.append(item["audio"].shape[-1])
        spectrogram_length.append(item["spectrogram"].shape[-1])
        text_encoded_length.append(item["text_encoded"].shape[-1])

    audio_padded = pad_sequence(audios, batch_first=True).permute(0, 2, 1)
    spectrogram_padded = pad_sequence(spectrograms, batch_first=True).permute(0, 2, 1)
    text_encoded_padded = pad_sequence(texts_encoded, batch_first=True).squeeze(-1)

    audio_length = torch.tensor(audio_length)
    spectrogram_length = torch.tensor(spectrogram_length)
    text_encoded_length = torch.tensor(text_encoded_length)
    
    result_batch = {
        "audio": audio_padded,
        "spectrogram": spectrogram_padded,
        "text": texts,
        "text_encoded": text_encoded_padded,
        "audio_path": audio_paths,
        "audio_length": audio_length,
        "spectrogram_length": spectrogram_length,
        "text_encoded_length": text_encoded_length
    }

    return result_batch
