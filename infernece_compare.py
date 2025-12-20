from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import librosa
import soundfile as sf

from models.unet_v2 import UNet as UNet_v2
from models.unet_v3 import UNet as UNet_v3
from models.unet_v4 import UNet as UNet_v4
from models.unet_torch import UNet as UNet_torch
from utils import *
from mk_dataset.dataset import *


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


# --- 오디오 변환 함수 ---
def convert_audio(magnitude, phase, n_fft=512, hop_length=160):
    # 데이터 타입 변경
    mag = magnitude.astype(np.float32)
    pha = phase.astype(np.float32)

    # 역정규화
    mag_db = (mag * 80.0) - 80.0

    # dB -> linear (dB를 다시 진폭으로 변환)
    # dB = 20 * log10(mag)
    mag_linear = 10 ** (mag_db / 20)

    # Complex Spectrogram 재생성
    stft_matrix = mag_linear * np.exp(1j * pha)

    # ISTFT
    audio = librosa.istft(stft_matrix, n_fft = n_fft, hop_length = hop_length)

    return audio


if __name__ == "__main__":
    # 1. 설정 (경로 및 하이퍼파라미터)
    # 저장된 체크포인트 파일 경로
    CHECKPOINT_PATH_2 = "./checkpoints/v2/last_checkpoint.pth"
    CHECKPOINT_PATH_3 = "./checkpoints/v3/checkpoint_35.pth"
    CHECKPOINT_PATH_4 = "./checkpoints/v4/last_checkpoint.pth"
    CHECKPOINT_PATH_TORCH = "./checkpoints/torch/last.pt"

    # 데이터 경로
    CLEAN_DIR = "./data/LibriSpeech/train-clean-100/"
    NOISE_DIR = "./data/noise_datasets/audio/"

    # 모델 설정 (학습 때와 동일해야 함)
    CHANNELS_2 = [1, 16, 32, 64, 128, 256]
    CHANNELS_3 = [1, 32, 64, 128, 256, 512]
    FILTER_SIZE = 3

    # 디바이스 설정 (PyTorch 모델용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Inference Device: {device}")

    # 2. 데이터셋 및 모델 준비
    test_dataset = NoiseRemovalDataset(CLEAN_DIR, NOISE_DIR, mode="val")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model_v2 = UNet_v2(CHANNELS_2, filter_size=FILTER_SIZE)
    model_v3 = UNet_v3(CHANNELS_3, filter_size=FILTER_SIZE)
    model_v4 = UNet_v4(CHANNELS_3, filter_size=FILTER_SIZE)
    model_torch = UNet_torch(CHANNELS_3).to(device)

    # 3. 가중치 로드
    load_custom_weights(model_v2, CHECKPOINT_PATH_2)
    load_custom_weights(model_v3, CHECKPOINT_PATH_3)
    load_custom_weights(model_v4, CHECKPOINT_PATH_4)

    ckpt_torch = torch.load(CHECKPOINT_PATH_TORCH, map_location=device)
    model_torch.load_state_dict(ckpt_torch["model_state_dict"])
    model_torch.eval()

    # 4. 추론 및 시각화
    # 랜덤한 샘플 하나 추출
    mixed_input, clean_target, mixed_phase = next(iter(test_loader))
    denoised_output_v2 = model_v2.forward(mixed_input)
    denoised_output_v3 = model_v3.forward(mixed_input)
    denoised_output_v4 = model_v4.forward(mixed_input)

    with torch.no_grad():
        denoised_output_torch = model_torch(mixed_input.to(device))

    # 텐서를 numpy로 변환 (시각화용)
    # shape: (1, 1, 256, 256) -> (256, 256)
    mixed_img = mixed_input.squeeze().numpy()
    output_img_v2 = denoised_output_v2.squeeze().numpy()
    output_img_v3 = denoised_output_v3.squeeze().numpy()
    output_img_v4 = denoised_output_v4.squeeze().numpy()
    output_img_torch = denoised_output_torch.cpu().squeeze().numpy()
    target_img = clean_target.squeeze().numpy()

    plt.rcParams['figure.dpi'] = 150
    # 1. Input (Noisy)
    plt.title("Input (Noisy Mixed)")
    plt.imshow(mixed_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 2. Output (Model Prediction)
    plt.title("Model Predicted_v2")
    plt.imshow(output_img_v2, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.title("Model Predicted_v3")
    plt.imshow(output_img_v3, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.title("Model Predicted_v4")
    plt.imshow(output_img_v4, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    plt.title("Model Predicted_torch")
    plt.imshow(output_img_torch, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Target (Clean)
    plt.title("Target (Clean Speech)")
    plt.imshow(target_img, aspect="auto", origin="lower", cmap="coolwarm")
    plt.colorbar(format="%+.1f dB")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 오디오로 변환
    mixed_phase = mixed_phase.squeeze().numpy()
    output_audio_noisy = convert_audio(mixed_img, mixed_phase)
    sf.write("./output_noisy.wav", output_audio_noisy, samplerate=16000)
    print("Saved : output_noisy.wav")

    output_audio_v2 = convert_audio(output_img_v2, mixed_phase)
    sf.write("./output_v2.wav", output_audio_v2, samplerate=16000)
    print("Saved : output_v2.wav")

    output_audio_v3 = convert_audio(output_img_v3, mixed_phase)
    sf.write("./output_v3.wav", output_audio_v3, samplerate=16000)
    print("Saved : output_v3.wav")

    output_audio_v4 = convert_audio(output_img_v4, mixed_phase)
    sf.write("./output_v4.wav", output_audio_v4, samplerate=16000)
    print("Saved : output_v4.wav")

    output_audio_torch = convert_audio(output_img_torch, mixed_phase)
    sf.write("./output_torch.wav", output_audio_torch, samplerate=16000)
    print("Saved : output_torch.wav")

    output_audio_clean = convert_audio(target_img, mixed_phase)
    sf.write("./output_clean.wav", output_audio_clean, samplerate=16000)
    print(f"Saved : {output_audio_clean}")



