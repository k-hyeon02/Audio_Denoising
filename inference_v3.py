import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from unet_v3 import UNet
from utils import *
from train_dataset.train_dataset import *


# --- 가중치 로드 함수 ---
def load_custom_weights(model, checkpoint_path):
    print(f"가중치 파일 로드 중: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        print("파일이 존재하지 않습니다.")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    for key, state_dict in checkpoint.items():
        try:
            parts = key.split("_")
            idx = int(parts[0])  # 모듈 인덱스
            if idx >= len(model.modules):
                continue

            target = model.modules[idx]  # enc1, down1, ...

            # DoubleConv 의 sub0/sub1 처리
            if len(parts) >= 3 and parts[1] == "DoubleConv":
                sub = parts[2]
                if sub == "sub0":
                    target = target.conv1
                elif sub == "sub1":
                    target = target.conv2

            # Conv2d, ConvTransposed2d 는 그대로 target 사용

            # 실제 가중치 주입
            for param_name, param_tensor in state_dict.items():
                if hasattr(target, param_name):
                    # 필요하면 clone() 해서 복사
                    setattr(target, param_name, param_tensor.clone())

        except Exception as e:
            # 디버깅 원하면 여기서 print(e, key) 찍어봐도 됨
            pass

    print("가중치 로드 완료!")


if __name__ == '__main__':
    # 1. 설정 (경로 및 하이퍼파라미터)
    # 저장된 체크포인트 파일 경로
    CHECKPOINT_PATH = "./checkpoints_3/last_checkpoint.pth"

    # 데이터 경로
    CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
    NOISE_DIR = "./data/noise_datasets/audio/"

    # 모델 설정 (학습 때와 동일해야 함)
    CHANNELS = [1, 32, 64, 128, 256, 512]
    FILTER_SIZE = 3

    # 2. 데이터셋 및 모델 준비
    test_dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR, mode="val")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = UNet(CHANNELS, filter_size=3)
    w_before = model.enc1.conv1.W.clone()
    load_custom_weights(model, CHECKPOINT_PATH)
    w_after = model.enc1.conv1.W
    print(torch.mean(torch.abs(w_after - w_before)))
    # 0이면 로드가 안됐다는거

    # 3. 추론 및 시각화
    # 랜덤한 샘플 하나 추출
    mixed_input, clean_target, _ = next(iter(test_loader))
    denoised_output = model.forward(mixed_input)

    # 텐서를 numpy로 변환 (시각화용)
    # shape: (1, 1, 256, 256) -> (256, 256)
    mixed_img = mixed_input.squeeze().numpy()
    output_img = denoised_output.squeeze().numpy()
    target_img = clean_target.squeeze().numpy()

    # 1. Input (Noisy)
    plt.subplot(1, 3, 1)
    plt.title("Input (Noisy Mixed)")
    plt.imshow(mixed_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    # 2. Output (Model Prediction)
    plt.subplot(1, 3, 2)
    plt.title("Output (Model Predicted)")
    plt.imshow(output_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    # 3. Target (Ground Truth)
    plt.subplot(1, 3, 3)
    plt.title("Target (Clean Speech)")
    plt.imshow(target_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    plt.show()
