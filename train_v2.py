import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import sys

from unet_v2 import UNet
from layers import *
from train_dataset.train_dataset import NoiseRemovalDataset

# 하이퍼파라미터 설정
LR = 0.001  
EPOCHS = 10  
BATCH_SIZE = 32  # 서버용 배치 사이즈

# 경로 설정 (본인의 환경에 맞게 수정 필요)
CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
NOISE_DIR = "./data/noise_datasets/audio/"
SAVE_DIR = "./checkpoints_2/" 

def get_device():
    """장치 자동 감지: Mac(MPS), NVIDIA(CUDA), CPU 순서"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()

# 학습 루프
def train():
    print(f"--- Training Start on {device} ---")

    # 1. 데이터셋
    # train data
    train_dataset = NoiseRemovalDataset(
        CLEAN_DIR, NOISE_DIR, mode='train', split_ratio=0.8
        )
    # validation data
    val_dataset = NoiseRemovalDataset(
        CLEAN_DIR, NOISE_DIR, mode='val', split_ratio=0.8
        )

    NUM_WORKERS = 24
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)

    print(f"Train Samples : {len(train_dataset)} | Val Samples : {len(val_dataset)}")

    # 2. 모델 생성
    model = UNet()

    # model을 device로 이동
    move_model_to_device(model, device)
    loss_func = MSELoss()

    # 로스 기록
    history = {'train_loss': [], 'val_loss': []}

    # 3. epoch 반복
    for epoch in range(EPOCHS):

        # [Train]
        train_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} : [Train]")

        for mixed, clean, _ in pbar:
            x = mixed.to(device)
            t = clean.to(device)

            # forward
            y = model.forward(x)

            # loss
            loss = loss_func.forward(y, t)

            # backward
            dout = loss_func.backward()
            model.backward(dout)

            # 가중치 업데이트
            model.step(lr=LR)

            # 기록
            train_loss_sum += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item(): .6f}"})

        avg_train_loss = train_loss_sum / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # [Validation]
        val_loss_sum = 0.0
        for mixed, clean, _ in val_loader:
            x = mixed.to(device)
            t = clean.to(device)

            # forward
            y = model.forward(x)
            loss = loss_func.forward(y, t)

            val_loss_sum += loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1} Result > Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # 5 epoch 마다 저장
        if (epoch+1) % 5 == 0:
            save_checkpoint(model, SAVE_DIR, f"checkpoint_{epoch+1}.pth")

        # 최종 체크포인트 저장
        save_checkpoint(model, SAVE_DIR, "last_checkpoint.pth")

    # 4. loss 시각화
    plt.plot(range(1, EPOCHS+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, EPOCHS + 1), history["val_loss"], label="Val Loss")
    plt.title("MSE Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    graph_path = os.path.join(SAVE_DIR, "loss_curve.png")
    plt.savefig(graph_path)
    plt.show()

if __name__ == "__main__":
    train()