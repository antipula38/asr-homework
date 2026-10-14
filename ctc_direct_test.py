import torch

checkpoint = torch.load(
    "saved/ema_smoke_test_4/model_best.pth",
    map_location="cpu",
    weights_only=False,
)

print("Checkpoint keys:", checkpoint.keys())
print("EMA weights saved:", "ema_state_dict" in checkpoint)
print("Model weights saved:", "state_dict" in checkpoint)
print("Epoch:", checkpoint["epoch"])