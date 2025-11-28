import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # 진행률 표시바
import os  

# PYTORCH_ENABLE_MPS_FALLBACK 환경 변수를 1로 설정합니다.
# 이렇게 하면 angle()처럼 지원 안 되는 연산은 CPU로 자동 전환됩니다.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from spectrogram import Spectrogram
from train_dataset import NoiseRemovalDataset
from unet import UNetDenoise

# --- 하이퍼파라미터 설정 (실험 조건) ---
BATCH_SIZE = 8  # 한 번에 학습할 데이터 수 (메모리 부족 시 줄이세요: 8 or 4)
LEARNING_RATE = 1e-4  # 학습률 (너무 크면 발산, 너무 작으면 느림)
EPOCHS = 20  # 전체 데이터셋 반복 횟수
NUM_WORKERS = 0  # 데이터 로딩에 사용할 CPU 코어 수
RESUME_FROM_EPOCH = 7  # 이어서 학습할 때 수정하는 변수 : 0이면 처음부터 시작, 3이면 epoch 3번 파일 불러와서 4번부터 시작

# 경로 설정 (본인의 환경에 맞게 수정 필요)
CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
NOISE_DIR = "./data/noise_datasets/audio/"
SAVE_DIR = "./checkpoints/"  # 모델 저장 경로


def get_device():
    """장치 자동 감지: Mac(MPS), NVIDIA(CUDA), CPU 순서"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train():
    # 1. 초기 설정
    local_start_epoch = RESUME_FROM_EPOCH
    device = get_device()
    print(f"🚀 학습 장치 설정: {device}")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 2. 데이터셋 및 로더 준비
    # (시료를 장비에 넣기 좋게 포장하는 과정)
    print("📂 데이터셋 로딩 중...")
    train_dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # 데이터를 잘 섞어야 학습이 잘 됨 (Ergodicity)
        num_workers=NUM_WORKERS,
        pin_memory=True,  # GPU 전송 속도 향상
    )
    print(f"✅ 총 데이터 개수: {len(train_dataset)}")

    spec_converter = Spectrogram().to(device)

    # 3. 모델, 손실함수, 옵티마이저 준비
    model = UNetDenoise().to(device)

    # 이어서 학습하기 로직
    if local_start_epoch > 0:
        checkpoint_path = os.path.join(SAVE_DIR, f"unet_epoch_{RESUME_FROM_EPOCH}.pth")

        if os.path.isfile(checkpoint_path):
            print(
                f"🔄 {RESUME_FROM_EPOCH}번 에폭 체크포인트를 불러옵니다: {checkpoint_path}"
            )
            # 저장된 가중치(Weight)를 모델에 덮어씌움
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            print(f"⚠️ 파일이 없습니다: {checkpoint_path}")
            print("처음부터 다시 시작합니다.")
            local_start_epoch = 0

    # Loss Function: MSE (Mean Squared Error)
    # 픽셀값(dB)의 차이를 제곱해서 평균 냄 -> 이걸 최소화하는 것이 목표
    criterion = nn.MSELoss()

    # Optimizer: Adam
    # 경사하강법(Gradient Descent)을 똑똑하게 수행하는 도구
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 학습 루프 (Training Loop)
    print("\n🔥 학습 시작!")
    model.train()  # 학습 모드 전환 (BN, Dropout 등 활성화)

    for epoch in range(RESUME_FROM_EPOCH, EPOCHS):
        running_loss = 0.0
        # tqdm으로 진행률 바 표시
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")

        for batch_idx, (mixed_wave, clean_wave) in enumerate(loop):
            # mixed_spec: 입력 (노이즈 낌)
            # clean_spec: 정답 (깨끗함)
            # _: 위상 정보는 학습 때는 필요 없음 (복원 때만 사용)

            # 데이터를 GPU(MPS)로 이동
            mixed_wave = mixed_wave.to(device)
            clean_wave = clean_wave.to(device)

            with torch.no_grad():
                mixed_spec,_ = spec_converter.to_spec(mixed_wave)
                clean_spec,_ = spec_converter.to_spec(clean_wave)

            # --- Forward Pass (예측) ---
            predictions = model(mixed_spec)

            # --- Compute Loss (오차 계산) ---
            loss = criterion(predictions, clean_spec)

            # --- Backward Pass (역전파 & 가중치 업데이트) ---
            optimizer.zero_grad()  # 이전 기울기 초기화
            loss.backward()  # 기울기 계산 (Gradient Calculation)
            optimizer.step()  # 가중치 수정 (Parameter Update)

            # --- Logging ---
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())  # 진행바에 현재 손실 표시

        # 에폭 종료 후 평균 손실 출력
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.6f}")

        # 모델 저장 (Checkpoint)
        # 나중에 학습 끊겨도 여기서부터 다시 하거나, 결과물 확인용
        torch.save(
            model.state_dict(), os.path.join(SAVE_DIR, f"unet_epoch_{epoch+1}.pth")
        )

    print("\n🎉 학습 완료! 모든 모델이 checkpoints 폴더에 저장되었습니다.")


if __name__ == "__main__":
    train()
