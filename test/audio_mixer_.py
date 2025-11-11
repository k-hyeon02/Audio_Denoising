import torch
import torchaudio
import random


class AudioMixer:
    def __init__(self, sr=16000, snr_range_db=(0, 15)):
        self.sr = sr
        self.snr_min = snr_range_db[0]
        self.snr_max = snr_range_db[1]
        self.resamplers = {}

    def preprocess_waveform(self, waveform, orig_sr):  # sampling_rate, ì±„ë„ ìˆ˜ í†µì¼
        if orig_sr != self.sr:
            if orig_sr not in self.resamplers:
                self.resamplers[orig_sr] = torchaudio.transforms.Resample(
                    orig_freq=orig_sr, new_freq=self.sr
                )
            waveform = self.resamplers[orig_sr](waveform)

        if waveform.shape[0] > 1:  # ì±„ë„ì´ 2ê°œ ì´ìƒì´ë©´
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        return waveform

    def _match_lengths(self, clean_wave, noise):
        clean_len = clean_wave.shape[-1]
        noise_len = noise.shape[-1]

        if clean_len > noise_len:  # ë…¸ì´ì¦ˆê°€ ë” ì§§ìœ¼ë©´: ë°˜ë³µ
            repeat_factor = clean_len // noise_len + 1
            # torch.catì„ ì‚¬ìš©í•˜ì—¬ [noise, noise, noise] í˜•íƒœë¡œ ì´ì–´ë¶™ìž„
            noise = torch.cat([noise] * repeat_factor, dim=-1)
            # ì •í™•í•œ ê¸¸ì´ë¡œ ìžë¥´ê¸°
            noise = noise[..., :clean_len]

        elif noise_len > clean_len:  # ë…¸ì´ì¦ˆê°€ ë” ê¸¸ë©´: ë¬´ìž‘ìœ„ë¡œ ìžë¥´ê¸°
            start_idx = random.randint(0, noise_len - clean_len)
            noise = noise[..., start_idx : start_idx + clean_len]

        return clean_wave, noise  # ê¸¸ì´ê°€ ê°™ì€ ê²½ìš°

    def _calculate_alpha(self, clean_wave, noise, target_snr_db):
        # ì „ë ¥(Power) = (ì§„í­ ì œê³±ì˜ í‰ê· ) + 1e-9 (0ìœ¼ë¡œ ë‚˜ëˆ„ê¸° ë°©ì§€)
        P_clean = torch.mean(clean_wave**2) + 1e-9
        P_noise = torch.mean(noise**2) + 1e-9

        # ëª©í‘œ SNR (dB -> ì„ í˜• ë¹„ìœ¨)
        target_ratio = 10 ** (target_snr_db / 10.0)

        # ìŠ¤ì¼€ì¼ë§ íŒ©í„°(alpha) ê³„ì‚°
        alpha = torch.sqrt(P_clean / (P_noise * target_ratio))

        return alpha

    def mix(self, clean_wave, noise_wave, target_snr_db=None):

        # 1. ì˜¤ë””ì˜¤ ê¸¸ì´ ë§žì¶”ê¸° (ê¹¨ë—í•œ ì˜¤ë””ì˜¤ ê¸°ì¤€)
        clean_wave, noise_wave = self._match_lengths(clean_wave, noise_wave)

        # 2. SNR ê°’ ê²°ì •
        if target_snr_db is None:
            used_snr_db = random.uniform(self.snr_min, self.snr_max)  # ë¬´ìž‘ìœ„ SNR ì„ íƒ
        else:
            used_snr_db = target_snr_db

        # 3. ë…¸ì´ì¦ˆ ìŠ¤ì¼€ì¼ë§ íŒ©í„°(alpha) ê³„ì‚°
        alpha = self._calculate_alpha(clean_wave, noise_wave, used_snr_db)

        # 4. ë…¸ì´ì¦ˆ ë³¼ë¥¨ ì¡°ì ˆ ë° ë¯¹ì‹±
        scaled_noise = alpha * noise_wave
        mixed_audio = clean_wave + scaled_noise

        return mixed_audio, clean_wave, used_snr_db


if __name__ == "__main__":
    # 1. ë°ì´í„° ë¡œë“œ
    clean_path = "./data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"

    clean_wave, clean_rate = torchaudio.load(clean_path)

    noise_path = "./data/noise_datasets/urbansound8k/audio/fold1/7061-6-0-0.wav"
    noise_wave, noise_rate = torchaudio.load(noise_path)

    # 2. AudioMixer ì¸ìŠ¤í„´ìŠ¤ ìƒì„±
    mixer = AudioMixer(sr=16000, snr_range_db=(0, 15))

    # 3. noise ì „ì²˜ë¦¬
    noise_wave = mixer.preprocess_waveform(noise_wave, noise_rate)

    # 4. ë¯¹ì‹±
    mixed_audio, clean_wave, snr_used = mixer.mix(
        clean_wave,
        noise_wave,
    )