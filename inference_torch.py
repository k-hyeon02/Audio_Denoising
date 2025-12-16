import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from unet_torch import UNet
from train_dataset.train_dataset import NoiseRemovalDataset

# --- 설정 (경로 및 하이퍼파라미터) ---
CHECKPOINT_PATH = "./checkpoints_torch/last.pt"

# 데이터 경로
CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
NOISE_DIR = "./data/noise_datasets/audio/"

# 모델 하이퍼파라미터
CHANNELS = [1, 32, 64, 128, 256, 512]


def inference():
    # 1. 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Inference Device: {device}")

    # 2. 데이터셋 및 로더 준비
    # 시각화를 위해 batch_size=1, shuffle=True로 설정하여 랜덤 샘플 확인
    test_dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR, mode="val")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # 3. 모델 초기화
    model = UNet(channels=CHANNELS).to(device)

    # 4. 가중치 로드
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        # train_torch.py에서 'model_state_dict' 키로 저장했으므로 해당 키를 불러옴
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # 만약 state_dict만 바로 저장된 경우 호환성 처리
            model.load_state_dict(checkpoint)

        print("Model weights loaded successfully!")
    else:
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    # 5. 추론 (Inference)
    model.eval()  

    with torch.no_grad():
        # 랜덤한 샘플 하나 추출
        mixed_input, clean_target, _ = next(iter(test_loader))

        # device로 이동
        mixed_input = mixed_input.to(device)

        # 모델 예측
        denoised_output = model(mixed_input)

    # 6. 시각화 (Tensor -> Numpy 변환)
    mixed_img = mixed_input.cpu().squeeze().numpy()
    output_img = denoised_output.cpu().squeeze().numpy()
    target_img = clean_target.squeeze().numpy()

    # 시각화 설정
    plt.figure(figsize=(15, 5))

    # 1. Input (Noisy)
    plt.subplot(1, 3, 1)
    plt.title("Input (Noisy Mixed)")
    plt.imshow(mixed_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.2f")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    # 2. Output (Model Prediction)
    plt.subplot(1, 3, 2)
    plt.title("Output (Model Predicted)")
    plt.imshow(output_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.2f")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    # 3. Target (Ground Truth)
    plt.subplot(1, 3, 3)
    plt.title("Target (Clean Speech)")
    plt.imshow(target_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.2f")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    inference()
