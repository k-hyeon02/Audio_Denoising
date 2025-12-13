import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

from unet_v2 import UNet as UNet_v2
from unet_v3 import UNet as UNet_v3
from unet_v4 import UNet as UNet_v4
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
                    setattr(target, param_name, param_tensor.clone())

        except Exception as e:
            print(e, key)
            pass

    print("가중치 로드 완료!")


if __name__ == "__main__":
    # 1. 설정 (경로 및 하이퍼파라미터)
    # 저장된 체크포인트 파일 경로
    CHECKPOINT_PATH_2 = "./checkpoints_2/last_checkpoint.pth"
    CHECKPOINT_PATH_3 = "./checkpoints_3/checkpoint_35.pth"
    CHECKPOINT_PATH_4 = "./checkpoints_4/last_checkpoint.pth"

    # 데이터 경로
    CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
    NOISE_DIR = "./data/noise_datasets/audio/"

    # 모델 설정 (학습 때와 동일해야 함)
    CHANNELS_2 = [1, 16, 32, 64, 128, 256]
    CHANNELS_3 = [1, 32, 64, 128, 256, 512]
    FILTER_SIZE = 3

    # 2. 데이터셋 및 모델 준비
    test_dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR, mode="val")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model_v2 = UNet_v2(CHANNELS_2, filter_size=FILTER_SIZE)
    model_v3 = UNet_v3(CHANNELS_3, filter_size=FILTER_SIZE)
    model_v4 = UNet_v4(CHANNELS_3, filter_size=FILTER_SIZE)

    w_before_v2 = model_v2.enc1.conv1.W.clone()
    w_before_v3 = model_v3.enc1.conv1.W.clone()
    w_before_v4 = model_v4.enc1.conv1.W.clone()

    load_custom_weights(model_v2, CHECKPOINT_PATH_2)
    load_custom_weights(model_v3, CHECKPOINT_PATH_3)
    load_custom_weights(model_v4, CHECKPOINT_PATH_4)

    w_after_v2 = model_v2.enc1.conv1.W
    print(torch.mean(torch.abs(w_after_v2 - w_before_v2)))

    w_after_v3 = model_v3.enc1.conv1.W
    print(torch.mean(torch.abs(w_after_v2 - w_before_v2)))

    w_after_v4 = model_v4.enc1.conv1.W
    print(torch.mean(torch.abs(w_after_v2 - w_before_v2)))
    # 0이면 로드가 안됐다는거

    # 3. 추론 및 시각화
    # 랜덤한 샘플 하나 추출
    mixed_input, clean_target, _ = next(iter(test_loader))
    denoised_output_v2 = model_v2.forward(mixed_input)
    denoised_output_v3 = model_v3.forward(mixed_input)
    denoised_output_v4 = model_v4.forward(mixed_input)

    # 텐서를 numpy로 변환 (시각화용)
    # shape: (1, 1, 256, 256) -> (256, 256)
    mixed_img = mixed_input.squeeze().numpy()
    output_img_v2 = denoised_output_v2.squeeze().numpy()
    output_img_v3 = denoised_output_v3.squeeze().numpy()
    output_img_v4 = denoised_output_v4.squeeze().numpy()
    target_img = clean_target.squeeze().numpy()

    # 1. Input (Noisy)
    plt.subplot(1, 5, 1)
    plt.title("Input (Noisy Mixed)")
    plt.imshow(mixed_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    # 2. Output (Model Prediction)
    plt.subplot(1, 5, 2)
    plt.title("Model Predicted_v2")
    plt.imshow(output_img_v2, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    plt.subplot(1, 5, 3)
    plt.title("Model Predicted_v3")
    plt.imshow(output_img_v3, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    plt.subplot(1, 5, 4)
    plt.title("Model Predicted_v4")
    plt.imshow(output_img_v4, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    # 3. Target (Ground Truth)
    plt.subplot(1, 5, 5)
    plt.title("Target (Clean Speech)")
    plt.imshow(target_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()
