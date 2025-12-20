# 'train-clean-100' 세트 다운로드 (약 6.3GB)
import torchaudio

dataset_librispeech = torchaudio.datasets.LIBRISPEECH(
    root="../data", url="train-clean-100", download=True
)

# UrbanSound8K 다운로드 (약 5.6GB)
import soundata

data_path = "../data/noise_datasets"
dataset_urbansound = soundata.initialize("urbansound8k", data_home=data_path)
dataset_urbansound.download()
dataset_urbansound.validate()