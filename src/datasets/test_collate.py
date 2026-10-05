from src.datasets.collate import collate_fn
from src.model.baseline_model import BaselineModel
from src.loss.ctc_loss import CTCLossWrapper
import torch

item1 = {
    "audio": torch.arange(100, dtype=torch.float32).unsqueeze(0),
    "spectrogram": torch.ones(1, 80, 11),
    "text": "cat",
    "text_encoded": torch.tensor([[3, 1, 20]], dtype=torch.long),
    "audio_path": "cat.wav",
}

item2 = {
    "audio": torch.arange(160, dtype=torch.float32).unsqueeze(0),
    "spectrogram": torch.ones(1, 80, 17) * 2,
    "text": "hello",
    "text_encoded": torch.tensor([[8, 5, 12, 12, 15]], dtype=torch.long),
    "audio_path": "hello.wav",
}

dataset_items = [item1, item2]

batch = collate_fn(dataset_items)
model = BaselineModel(
    n_feats=80,
    n_tokens=30
)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)
first_param = next(model.parameters())
before = first_param.detach().clone()

# training step
for step in range(200):
    optimizer.zero_grad()
    result = model(
        spectrogram=batch["spectrogram"],
        spectrogram_length=batch["spectrogram_length"]
    )
    ctc_loss = CTCLossWrapper()
    loss_result = ctc_loss(
        result["log_probs"],
        result["log_probs_length"],
        batch["text_encoded"],
        batch["text_encoded_length"]
    )
    loss_result["loss"].backward()
    optimizer.step()
    if step % 20 == 0:
        print(step, loss_result)

after = first_param.detach().clone()



difference = (after - before).abs().sum()
print(difference)

