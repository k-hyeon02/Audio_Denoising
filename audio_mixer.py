import torch
import torchaudio
import random

# sampling rate = 16000Hz : 1초에 16000번 녹음
# 1번 녹음하는데 걸리는 시간 = 1/16000초
# 160 간격 = 0.01초 = 10ms
# target frame = 256 : 스펙토그램의 가로축을 2.55초, 256칸으로 만들기 위함 -> 학습 데이터 규격 통일

class AudioMixer:
    def __init__(self, sr=16000, target_frame=256, hop_length=160):
        self.sr = sr
        self.target_samples = (target_frame-1) * hop_length  # 2.55초 오디오 = 40,800 샘플 -> 256 프레임
        self.resamplers = {}

    def _preprocess_waveform(self, waveform, orig_sr):  # sampling_rate, 채널 수 통일
        if orig_sr != self.sr:
            if orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=self.sr
                )
            waveform = self.resamplers[orig_sr](waveform)

        if waveform.shape[0] > 1:  # 채널이 2개 이상이면
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def _prepare_segment(self, waveform, is_noise=False):
        '''
        2.55초 보다 짧은 오디오 파일은 zero padding
        2.55초 보다 긴 오디오는 랜덤으로 2.55초 구간 선택
        노이즈 오디오 ??
        '''
        current_len = waveform.shape[-1]
        target_len = self.target_samples

        if current_len > self.target_samples:
            start = random.randint(0, current_len - self.target_samples)
            waveform = waveform[..., start : start + self.target_samples]

        elif current_len < target_len:
            if is_noise:  # 노이즈는 반복
                repeat_factor = target_len // current_len + 1
                waveform = torch.cat([waveform] * repeat_factor, dim=1)
                waveform = waveform[..., :target_len]  # 2.55s 넘어가는 부분 자름
            else:
                pad_amount = target_len - current_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

        return waveform

    def _calculate_alpha(self, clean_wave, noise, target_snr_db):
        # 전력(Power) = (진폭 제곱의 평균) + 1e-9 (0으로 나누기 방지)
        P_clean = torch.mean(clean_wave**2) + 1e-9
        P_noise = torch.mean(noise**2) + 1e-9

        # 목표 SNR (dB -> 선형 비율)
        target_ratio = 10 ** (target_snr_db / 10.0)

        # 스케일링 팩터(alpha) 계산
        alpha = torch.sqrt(P_clean / (P_noise * target_ratio))

        return alpha

    def mix(self, clean_wave, noise_wave, clean_rate, noise_rate, snr_db):

        # 1. 전처리 (sampling rate, 채널 수 통일)
        clean_wave = self._preprocess_waveform(clean_wave, clean_rate)
        noise_wave = self._preprocess_waveform(noise_wave, noise_rate)

        # 2. 규격화 (40,800 samples)
        clean_segment = self._prepare_segment(clean_wave, is_noise=False)
        noise_segment = self._prepare_segment(noise_wave, is_noise=True)

        # 3. 노이즈 스케일링 팩터(alpha) 계산
        P_clean = torch.mean(clean_segment**2) + 1e-9
        P_noise = torch.mean(noise_segment**2) + 1e-9

        target_ratio = 10 ** (snr_db/10)
        alpha = torch.sqrt(P_clean / (P_noise * target_ratio))

        # 4. 중첩
        mixed_wave = clean_segment + (alpha * noise_segment)

        return mixed_wave, clean_segment


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
        clean_wave, noise_wave,
        clean_rate, noise_rate,
        snr_db=test_snr
    )

    print(f"Input Clean Shape: {clean_wave.shape}")
    print(f"Input Noise Shape: {noise_wave.shape}")
    print(f"Output Mixed Shape: {mixed_audio.shape}")  # (1, 40800) 나와야 성공
    print(f"Output Target Shape: {clean_segment.shape}")  # (1, 40800) 나와야 성공
    print(f"Applied SNR: {test_snr:.2f} dB")
