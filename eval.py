import torch
from unet_v4 import UNet
from train_dataset.train_data import NoiseRemovalDataset

CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
NOISE_DIR = "./data/noise_datasets/audio/"

device = "mps" if torch.cuda.is_available() else "cpu"

model = UNet([1,32,64,128,256,512], device=device).to(device)

ckpt = torch.load("./saved/2/checkpoints/checkpoint_50.pth", map_location=device)
model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
model.eval()

dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR, mode="val", split_ratio=0.8)
mixed, clean, _ = dataset[0]
print(mixed.shape)

x = mixed.unsqueeze(0).to(device)

with torch.no_grad():
    y = model(x)

print("Done. Output shape:", y.shape)

import matplotlib.pyplot as plt

# img: torch.Tensor, shape (H, W) 또는 (1, H, W)
img = y.squeeze().cpu()   # (H, W)

# plt.figure()
# plt.imshow(img, cmap="gray")
# plt.colorbar()
# plt.axis("off")
# plt.show()


fig, axs = plt.subplots(1, 3, figsize=(12, 4))

axs[0].imshow(mixed[0].cpu(), cmap="gray")
axs[0].set_title("Noisy")
axs[0].axis("off")

axs[1].imshow(clean[0].cpu(), cmap="cmap")
axs[1].set_title("Clean")
axs[1].axis("off")

axs[2].imshow(y[0, 0].cpu(), cmap="gray")
axs[2].set_title("Output")
axs[2].axis("off")

plt.tight_layout()
plt.show()
