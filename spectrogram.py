import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import random
import matplotlib.pyplot as plt


class Spectrogram:
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length

        self.spec_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
        self.melspec_transform = T.MelSpectrogram(n_fft=n_fft, hop_length=hop_length)

        self.amplitude_to_dB = T.AmplitudeToDB(stype="power")

    def spectrogram(self, waveform):
        spec = self.spec_transform(waveform)
        dB_spec = self.amplitude_to_dB(spec)

        return dB_spec

    def mel_spectrogram(self, waveform):
        mel_spec = self.melspec_transform(waveform)
        dB_mel_spec = self.amplitude_to_dB(mel_spec)

        return dB_mel_spec

    def plot(self, dB_spec, ax, title):
        spec_visualize = dB_spec[0].numpy()
        img = librosa.display.specshow(
            spec_visualize,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            x_axis="time",
            y_axis="linear",
            ax=ax,
        )
        ax.set_title(title)

        return img


if __name__ == "__main__":
    spec_converter = Spectrogram()

    # clean
    clean_path = "./data/LibriSpeech/train-clean-100/19/198/19-198-0000.flac"
    clean_wave, clean_rate = torchaudio.load(clean_path)

    spec_clean = spec_converter.spectrogram(clean_wave)
    mel_spec_clean = spec_converter.mel_spectrogram(clean_wave)

    # noise
    noise_path = "./data/noise_datasets/urbansound8k/audio/fold1/7061-6-0-0.wav"
    noise_wave, noise_rate = torchaudio.load(noise_path)

    # sampling rate 통일
    noise_wave = torchaudio.transforms.Resample(
        orig_freq=noise_rate, new_freq=clean_rate
    )(noise_wave)

    # 채널 수 통일
    noise_wave = torch.mean(noise_wave, dim=0, keepdim=True)

    # 길이 통일
    clean_len = clean_wave.shape[-1]
    noise_len = noise_wave.shape[-1]

    if clean_len > noise_len:
        repeat_factor = clean_len // noise_len + 1
        noise_wave = torch.cat([noise_wave] * repeat_factor, dim=1)
        noise_wave = noise_wave[..., :clean_len]

    elif clean_len < noise_len:
        start_idx = random.randint(0, noise_len - clean_len)
        noise_wave = noise_wave[..., start_idx : start_idx + clean_len]

    else:
        clean_wave, noise_wave

    spec_noise = spec_converter.spectrogram(noise_wave)
    mel_spec_noise = spec_converter.mel_spectrogram(noise_wave)

    # mixed
    mixed_path = "./test/mixed_audio_example.flac"
    mixed_wave, mixed_rate = torchaudio.load(mixed_path)
    spec_mixed = spec_converter.spectrogram(mixed_wave)
    mel_spec_mixed = spec_converter.mel_spectrogram(mixed_wave)

    fig, axes = plt.subplots(nrows=1, ncols=3)
    plt.tight_layout()
    img1 = spec_converter.plot(mel_spec_clean, axes[0], "Clean Mel Spectorgram")
    img2 = spec_converter.plot(mel_spec_noise, axes[1], "Noise Mel Spectorgram")
    img3 = spec_converter.plot(mel_spec_mixed, axes[2], "Mixed Mel Spectorgram")

    cbar = fig.colorbar(img1, ax=axes)
    cbar.set_label("Amplitutde(dB)")

    plt.show()
