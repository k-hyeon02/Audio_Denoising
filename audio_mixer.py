import torch
import torchaudio
import random


class AudioMixer:
    def __init__(self, sr=16000, snr_range_db=(0, 15)):
        self.sr = sr
        self.snr_min = snr_range_db[0]
        self.snr_max = snr_range_db[1]
        self.resamplers = {}

    def preprocess_waveform(self, waveform, orig_sr):  # sampling_rate, 채널 수 통일
        if orig_sr != self.sr:
            if orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=self.sr
                )
            waveform = self.resamplers[orig_sr](waveform)

        if waveform.shape[0] > 1:  # 채널이 2개 이상이면
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def _match_lengths(self, clean_wave, noise):
        clean_len = clean_wave.shape[-1]
        noise_len = noise.shape[-1]

        if clean_len > noise_len:  # 노이즈가 더 짧으면: 반복
            repeat_factor = clean_len // noise_len + 1
            # torch.cat을 사용하여 [noise, noise, noise] 형태로 이어붙임
            noise = torch.cat([noise] * repeat_factor, dim=-1)
            # 정확한 길이로 자르기
            noise = noise[..., :clean_len]

        elif noise_len > clean_len:  # 노이즈가 더 길면: 무작위로 자르기
            start_idx = random.randint(0, noise_len - clean_len)
            noise = noise[..., start_idx : start_idx + clean_len]

        return clean_wave, noise  # 길이가 같은 경우

    def _calculate_alpha(self, clean_wave, noise, target_snr_db):
        # 전력(Power) = (진폭 제곱의 평균) + 1e-9 (0으로 나누기 방지)
        P_clean = torch.mean(clean_wave**2) + 1e-9
        P_noise = torch.mean(noise**2) + 1e-9

        # 목표 SNR (dB -> 선형 비율)
        target_ratio = 10 ** (target_snr_db / 10.0)

        # 스케일링 팩터(alpha) 계산
        alpha = torch.sqrt(P_clean / (P_noise * target_ratio))

        return alpha

    def mix(self, clean_wave, noise_wave, target_snr_db=None):

        # 1. 오디오 길이 맞추기 (깨끗한 오디오 기준)
        clean_wave, noise_wave = self._match_lengths(clean_wave, noise_wave)

        # 2. SNR 값 결정
        if target_snr_db is None:
            used_snr_db = random.uniform(self.snr_min, self.snr_max)  # 무작위 SNR 선택
        else:
            used_snr_db = target_snr_db

        # 3. 노이즈 스케일링 팩터(alpha) 계산
        alpha = self._calculate_alpha(clean_wave, noise_wave, used_snr_db)

        # 4. 노이즈 볼륨 조절 및 믹싱
        scaled_noise = alpha * noise_wave
        mixed_audio = clean_wave + scaled_noise

        return mixed_audio, clean_wave, used_snr_db


if __name__ == "__main__":
    # 1. 데이터 로드
    clean_path = "./data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

    clean_wave, clean_rate = torchaudio.load(clean_path)

    noise_path = "./data/noise_datasets/urbansound8k/audio/fold1/7061-6-0-0.wav"
    noise_wave, noise_rate = torchaudio.load(noise_path)

    # 2. AudioMixer 인스턴스 생성
    mixer = AudioMixer(sr=16000, snr_range_db=(0, 15))

    # 3. noise 전처리
    noise_wave = mixer.preprocess_waveform(noise_wave, noise_rate)

    # 4. 믹싱
    mixed_audio, clean_wave, snr_used = mixer.mix(
        clean_wave,
        noise_wave,
    )
