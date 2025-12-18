import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

from unet_v4 import UNet
from layers import *
from utils import *
from train_dataset.train_data import NoiseRemovalDataset

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# 하이퍼파라미터 설정
LR = 0.0001
EPOCHS = 50
BATCH_SIZE = 10  # 서버용 배치 사이즈

# 경로 설정 (본인의 환경에 맞게 수정 필요)
CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
NOISE_DIR = "./data/noise_datasets/audio/"
SAVE_DIR = "./saved"

num_dirs = sum(
    1 for f in os.listdir(SAVE_DIR)
    if os.path.isdir(os.path.join(SAVE_DIR, f))
)

RUN_NUM = str(num_dirs)

# 경로 구성
RUN_DIR = os.path.join(SAVE_DIR, RUN_NUM)
CHECK_DIR = os.path.join(RUN_DIR, "checkpoints")
LOG_PATH = os.path.join(RUN_DIR, "training_log.csv")

# 디렉토리 생성
os.makedirs(CHECK_DIR, exist_ok=True)

# 로그 파일 초기화
init_log_file(LOG_PATH)

device = get_device()


# 학습 루프
def train():
    print(f"--- Training Start on {device} ---")

    # 1. 데이터셋
    # train data
    train_dataset = NoiseRemovalDataset(
        CLEAN_DIR, NOISE_DIR, mode="train", split_ratio=0.8
    )
    # validation data
    val_dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR, mode="val", split_ratio=0.8)

    NUM_WORKERS = 24
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train Samples : {len(train_dataset)} | Val Samples : {len(val_dataset)}")

    # 2. 모델 생성
    model = UNet(channels=[1, 64, 128, 256, 512, 1024], device=device)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # model을 device로 이동

    # loss_func = F.l1_loss


    # 로스 기록
    history = {"train_loss": [], "val_loss": [], "train_psnr": [], "val_psnr": []}

    # 3. epoch 반복
    for epoch in range(EPOCHS):

        # [Train]
        train_loss_sum = 0.0
        train_psnr_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} : [Train]")

        for mixed, clean, _ in pbar:
            x = mixed.to(device, non_blocking=True)
            t = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                y = model(x)
                loss = stft_loss(y, t) + 0.5 * F.l1_loss(y, t)



            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # PSNR 계산
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

        for mixed, clean, _ in val_loader:
            x = mixed.to(device, non_blocking=True)
            t = clean.to(device, non_blocking=True)

            with autocast():
                y = model(x)
                loss = stft_loss(y, t) + 0.5 * F.l1_loss(y, t)

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

        # 5 epoch 마다 저장
        log_epoch_metrics(
            LOG_PATH,
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
            avg_train_psnr,
            avg_val_psnr
        )

        save_checkpoint(model, CHECK_DIR, f"checkpoint_{epoch+1}.pth")

    # # 4. loss 시각화
    # plt.plot(range(1, EPOCHS + 1), history["train_loss"], label="Train Loss")
    # plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Val Loss")
    # plt.title("MSE Loss")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.grid(True)

    # loss_g_path = os.path.join(SAVE_DIR, "loss_curve.png")
    # plt.savefig(loss_g_path)
    # plt.show()

    # # 5. PSNR 시각화
    # plt.plot(range(1, EPOCHS + 1), history["train_psnr"], label="Train PSNR")
    # plt.plot(range(1, EPOCHS + 1), history["val_psnr"], label="Val PSNR")
    # plt.title("PSNR")
    # plt.xlabel("Epoch")
    # plt.ylabel("PSNR")
    # plt.legend()
    # plt.grid(True)

    # psnr_g_path = os.path.join(SAVE_DIR, "psnr_curve.png")
    # plt.savefig(psnr_g_path)
    # plt.show()


if __name__ == "__main__":
    train()
