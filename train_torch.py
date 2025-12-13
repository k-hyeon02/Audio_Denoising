import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys
import torch.nn as nn

from unet_torch import UNet
from utils import *
from train_dataset.train_dataset import NoiseRemovalDataset

# 하이퍼파라미터 설정
LR = 0.0001
EPOCHS = 50
BATCH_SIZE = 16  # 서버용 배치 사이즈
NUM_WORKERS = 24


# 경로 설정 (본인의 환경에 맞게 수정 필요)
CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
NOISE_DIR = "./data/noise_datasets/audio/"
SAVE_DIR = "./checkpoints_torch/"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def save_torch_checkpoint(model, optimizer, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


# 학습 루프
def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"--- Training Start on {device} ---")

    # 1. 데이터셋
    # train data
    train_dataset = NoiseRemovalDataset(
        CLEAN_DIR, NOISE_DIR, mode="train", split_ratio=0.8
    )
    # validation data
    val_dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR, mode="val", split_ratio=0.8)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Train Samples : {len(train_dataset)} | Val Samples : {len(val_dataset)}")

    # 2. 모델 생성
    model = UNet(channels=[1, 32, 64, 128, 256, 512]).to(device)

    # model을 device로 이동
    loss_func = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 로스 기록
    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}

    # 3. epoch 반복
    for epoch in range(EPOCHS):

        # [Train]
        model.train()
        train_loss_sum = 0.0
        train_psnr_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} : [Train]")
        for mixed, clean, _ in pbar:
            x = mixed.to(device)
            t = clean.to(device)

            optimizer.zero_grad()

            # forward
            y = model(x)

            # loss
            loss = loss_func(y, t)

            # backward
            loss.backward()

            # 가중치 업데이트
            optimizer.step()
            with torch.no_grad():
                psnr = calculate_psnr(y, t)

            # 기록
            train_loss_sum += loss.item()
            train_psnr_sum += psnr.item()
            pbar.set_postfix(
                {"Loss": f"{loss.item(): .6f}", "PSNR": f"{psnr.item(): .2f}"}
            )

        avg_train_loss = train_loss_sum / len(train_loader)
        avg_train_psnr = train_psnr_sum / len(train_loader)

        history["train_loss"].append(avg_train_loss)
        history["train_psnr"].append(avg_train_psnr)

        # [Validation]
        model.eval()
        val_loss_sum = 0.0
        val_psnr_sum = 0.0

        with torch.no_grad():
            for mixed, clean, _ in val_loader:
                x = mixed.to(device)
                t = clean.to(device)

                # forward
                y = model(x)
                loss = loss_func(y, t)
                psnr = calculate_psnr(y, t)

                val_loss_sum += loss.item()
                val_psnr_sum += psnr.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        avg_val_psnr = val_psnr_sum / len(val_loader)

        history["val_loss"].append(avg_val_loss)
        history["val_psnr"].append(avg_val_psnr)

        print(
            f"Epoch {epoch+1} Result > "
            f"Loss: {avg_train_loss:.6f} (Val: {avg_val_loss:.6f}) | "
            f"PSNR: {avg_train_psnr:.2f}dB (Val: {avg_val_psnr:.2f}dB)"
        )

        # epoch 마다 저장
        save_torch_checkpoint(model, optimizer, SAVE_DIR, "last.pt")
        if (epoch + 1) % 5 == 0:
            save_torch_checkpoint(model, optimizer, SAVE_DIR, f"epoch_{epoch+1}.pt")

    # 4. loss 시각화
    plt.plot(range(1, EPOCHS + 1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Val Loss")
    plt.title("L1 Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_g_path = os.path.join(SAVE_DIR, "loss_curve.png")
    plt.savefig(loss_g_path)
    plt.show()

    # 5. PSNR 시각화
    plt.plot(range(1, EPOCHS + 1), history["train_psnr"], label="Train PSNR")
    plt.plot(range(1, EPOCHS + 1), history["val_psnr"], label="Val PSNR")
    plt.title("PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.legend()
    plt.grid(True)

    psnr_g_path = os.path.join(SAVE_DIR, "psnr_curve.png")
    plt.savefig(psnr_g_path)
    plt.show()


if __name__ == "__main__":
    train()
