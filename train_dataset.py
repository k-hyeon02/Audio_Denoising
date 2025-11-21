from torch.utils.data import Dataset
import torchaudio
import glob
import random
import os

from audio_mixer import AudioMixer
from spectrogram import Spectrogram

class NoiseRemovalDataset(Dataset):
    def __init__(self, clean_dir, noise_dir, target_frame=256, hop_length=160):

        # 모든 파일 경로 리스트업
        self.clean_files = sorted(
            glob.glob(os.path.join(clean_dir, "**/*.flac"), recursive=True)
        )
        self.noise_files = sorted(
            glob.glob(os.path.join(noise_dir, "**/*.wav"), recursive=True)
        )

        # 도구(Tools) 초기화
        self.mixer = AudioMixer(target_frame=target_frame, hop_length=hop_length)
        self.processor = Spectrogram(n_fft=512, hop_length=hop_length)

    def __len__(self):
        # 전체 학습 데이터 수 (클린 파일 개수 기준)
        return len(self.clean_files)

    def __getitem__(self, idx):
        # 1. 파일 로드 및 메타데이터 획득
        clean_path = self.clean_files[idx]
        noise_path = random.choice(self.noise_files)  # 노이즈는 랜덤 선택 (데이터 증강)

        clean_wave, clean_sr = torchaudio.load(clean_path)
        noise_wave, noise_sr = torchaudio.load(noise_path)

        # 2. 랜덤 SNR 결정 (0dB ~ 15dB 사이에서 무작위 선택)
        snr = random.uniform(0, 15)

        # 3. 믹싱 (AudioMixer 사용)
        # 규격화(40,800 샘플)와 믹싱이 동시에 수행됨
        mixed_wave, clean_target_wave = self.mixer.mix(
            clean_wave, noise_wave, clean_sr, noise_sr, snr_db=snr
        )

        # 4. 스펙트로그램 변환
        # 결과: (1, 256, 256) 텐서
        mixed_spec = self.processor.to_spec(mixed_wave)
        clean_spec = self.processor.to_spec(clean_target_wave)

        # 5. 모델 입력/정답 반환
        return mixed_spec, clean_spec


if __name__ == "__main__":
    clean_dir = "./data/LibriSpeech/train-clean-100/"
    noise_dir = "./data/noise_datasets/audio/"

    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import librosa.display

    try:
        # 2. 데이터셋 인스턴스 생성
        dataset = NoiseRemovalDataset(clean_dir, noise_dir)
        print(f"✅ 데이터셋 생성 완료! 총 데이터 수: {len(dataset)}개")

        # 3. 데이터 로더 연결 (배치 테스트)
        # 배치 사이즈 4로 설정하여 4개씩 묶어서 나오는지 확인
        BATCH_SIZE = 4
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 4. 첫 번째 배치 추출 (Iterate)
        mixed_batch, clean_batch = next(iter(loader))

        # 5. 차원(Shape) 검증
        print("\n--- [1] 차원(Shape) 검증 ---")
        print(f"Mixed Batch Shape: {mixed_batch.shape}")
        print(f"Clean Batch Shape: {clean_batch.shape}")

        # 검증 로직: (Batch, Channel, Freq, Time) = (4, 1, 256, 256) 이어야 함
        expected_shape = (BATCH_SIZE, 1, 256, 256)
        if mixed_batch.shape == expected_shape and clean_batch.shape == expected_shape:
            print("Testing Dimensions... PASS (규격이 정확합니다)")
        else:
            print(f"Testing Dimensions... FAIL (기대값: {expected_shape})")

        # 6. 시각적(Visual) 검증
        print("\n--- [2] 시각적 검증 (첫 번째 샘플) ---")
        import librosa.display  # 라이브러리 임포트 필요

        # 배치 중 첫 번째 데이터만 꺼내서 그리기
        mixed_sample = mixed_batch[0].squeeze().numpy()  # (256, 256)
        clean_sample = clean_batch[0].squeeze().numpy()  # (256, 256)

        plt.figure(figsize=(12, 5))

        # [수정] plt.imshow 대신 librosa.display.specshow 사용
        plt.subplot(1, 2, 1)
        plt.title("Mixed Input (Noisy)")
        librosa.display.specshow(
            mixed_sample,
            sr=16000,
            hop_length=160,
            x_axis="time",
            y_axis="linear", 
            cmap="magma",
        )
        plt.colorbar(format="%+2.0f dB")

        plt.subplot(1, 2, 2)
        plt.title("Clean Target (Ground Truth)")
        librosa.display.specshow(
            clean_sample,
            sr=16000,
            hop_length=160,
            x_axis="time",
            y_axis="linear",
            cmap="magma",
        )
        plt.colorbar(format="%+2.0f dB")

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        print("파일 경로가 정확한지, 데이터가 해당 폴더에 있는지 확인해주세요.")
