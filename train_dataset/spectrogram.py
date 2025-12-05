import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import random
import matplotlib.pyplot as plt
from train_dataset.audio_mixer import AudioMixer

# 16000Hz에서 25ms = 400 < 512 = n_fft 로 설정
# 이동 간격 : hop_length = 160 : 16000Hz에서 10ms, 데이터 길이가 40,800이므로 총 0~255 프레임의 시간축 생성
class Spectrogram:
    def __init__(self, n_fft=512, hop_length=160):
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)

        self.amplitude_to_dB = T.AmplitudeToDB(stype="power")

    def to_spec(self, waveform):
        spec = self.spec_transform(waveform)

        # 진폭 추출 및 dB 변환
        magnitude = spec.abs()
        power = magnitude ** 2
        dB_spec = self.amplitude_to_dB(power)
        dB_spec = dB_spec[..., :-1, :]  # 세로축(주파수)를 257 -> 256 : 짝수로 맞추기 위해 가장 높은 주파수 bin 하나 제거

        phase = spec.angle()
        phase = phase[..., :-1, :]

        return dB_spec, phase

    def to_melspec(self, waveform):
        melspec = self.melspec_transform(waveform)
        dB_melspec = self.amplitude_to_dB(melspec)
        dB_melspec = dB_melspec[..., :-1, :]

        return dB_melspec

    def plot(self, dB_spec, ax, title):
        spec_visualize = dB_spec[0].numpy()
        img = librosa.display.specshow(
            spec_visualize,
            sr=16000,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="linear",
            ax=ax,
            cmap="magma"
        )
        ax.set_title(title)

        return img


if __name__ == "__main__":
    # 1. 데이터 로드
    clean_path = "./data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

    clean_wave, clean_rate = torchaudio.load(clean_path)

    noise_path = "./data/noise_datasets/audio/fold1/7061-6-0-0.wav"
    noise_wave, noise_rate = torchaudio.load(noise_path)

    # 2. AudioMixer 인스턴스 생성
    mixer = AudioMixer(sr=16000, target_frame=256, hop_length=160)

    # 3. 믹싱
    test_snr = random.uniform(0, 15)

    mixed_audio, clean_segment = mixer.mix(
        clean_wave, noise_wave, clean_rate, noise_rate, snr_db=test_snr
    )

    # 4. Spectrogram 인스턴스 생성
    spec = Spectrogram()

    fig, axes = plt.subplots(nrows=1, ncols=2)
    plt.tight_layout()

    clean_dB_spec = spec.to_spec(clean_segment)
    img_clean = spec.plot(clean_dB_spec, axes[0], title='Clean Segment')
    fig.colorbar(img_clean, ax=axes[0], format='%+2.0f dB') 

    mixed_dB_spec = spec.to_spec(mixed_audio)
    img_mixed = spec.plot(mixed_dB_spec, axes[1], title="Mixed Audio")
    fig.colorbar(img_mixed, ax=axes[1], format="%+2.0f dB")  

    plt.show()
