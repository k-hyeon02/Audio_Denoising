# Audio_Denoising

---
2025-2학기 <딥러닝> 1조 프로젝트

## 📖 Project Overview
이 프로젝트는 **U-Net** 아키텍처를 기반으로 한 **오디오 노이즈 제거(Audio Denoising)** 딥러닝 모델 구현체입니다.

가장 큰 특징은 PyTorch와 같은 고수준 프레임워크의 자동 미분 기능에 전적으로 의존하지 않고, **torch 연산만을 사용하여 딥러닝의 핵심 레이어와 최적화 알고리즘(Optimizer)을 밑바닥부터(From Scratch) 직접 구현**했다는 점입니다. 이를 통해 딥러닝의 내부 동작 원리와 역전파(Backpropagation) 과정을 깊이 있게 이해하고 검증하였습니다.  
비교 검증을 위해 PyTorch 버전의 구현체(`*_torch.py`)도 함께 포함되어 있습니다.

---
## 🚀 Key Features (핵심 구현 내용)

### 1. Deep Learning Core Implementation (From Scratch)
`torch.nn`이나 `torch.optim`을 사용하지 않고 딥러닝의 핵심 요소를 직접 구현했습니다.
* **Vectorized im2col/col2im:** for문 반복을 최소화하고 `broadcasting`을 활용한 고속 컨볼루션 연산 구현.
* **Backpropagation:** Convolution, Transposed Conv, Activation 등 모든 레이어의 역전파 수식 직접 유도 및 구현.
* **Custom Optimizer:**
    * **SGD (v2, v3):** 기본적인 확률적 경사 하강법.
    * **Adam (v4):** Momentum(m)과 RMSProp(v)을 결합한 Adam Optimizer의 수식을 직접 코드로 구현하여 학습 안정성 확보.

### 2. Robust Data Pipeline (`train_dataset/`)
정해진 데이터셋을 사용하는 것이 아니라, 학습 시점에 실시간으로 데이터를 생성하여 일반화 성능을 높였습니다.
* **On-the-fly Mixing:** 깨끗한 음성(LibriSpeech)에 소음(UrbanSound8K)을 **랜덤한 SNR(0~15dB)** 비율로 실시간 합성.
* **Spectrogram Processing:** STFT(Short-Time Fourier Transform)를 통해 Time-Frequency 도메인으로 변환 및 정규화.

---
## 🏗️ Model Architecture: U-Net

본 프로젝트에서 사용한 **U-Net**은 본래 의료 영상 분할(Segmentation)을 위해 제안되었으나, 오디오 스펙트로그램의 **잡음 제거(Denoising)** 작업에서도 탁월한 성능을 보입니다.

### 🔍 핵심 구조 (Key Components)

1.  **Contracting Path (Encoder):**
    * 입력 스펙트로그램의 특징(Feature)을 추출하고 압축합니다.
    * `Conv2d`와 `Pooling`을 거치며 이미지 크기는 줄어들고(Downsampling), 채널(특징) 수는 늘어납니다.
2.  **Bottleneck:**
    * 가장 압축된 형태의 정보(Latent Vector)를 담고 있는 구간입니다.
3.  **Expansive Path (Decoder):**
    * 압축된 정보를 다시 원래 스펙트로그램 크기로 복원합니다.
    * `Transposed Conv`를 사용하여 이미지를 확대(Upsampling)합니다.
4.  **Skip Connection (Concatenation):**
    * **U-Net의 핵심**입니다. 인코더에서 줄어들면서 손실된 **공간적 정보(Spatial Information)**를 디코더에 바로 전달합니다.
    * 이를 통해 노이즈는 제거하면서도 목소리의 선명한 구조(주파수 디테일)는 유지할 수 있습니다.

### 📐 구조도 (Diagram)

입력은 `(Channel=1, 256, 256)` 크기의 오디오 스펙트로그램이며, 출력은 노이즈가 제거된 동일한 크기의 스펙트로그램입니다.

```text
Input (1, 256, 256)
     │
     ▼
[Encoder 1] ──────────────────────────(Skip Connection)──────────────────────────► [Decoder 1] ──► Sigmoid ──► Output
(64, 128, 128)                                                                    (64, 128, 128)      
     │                                                                                  │             
     ▼                                                                                  ▲             
[Encoder 2] ───────────────────(Skip Connection)───────────────────► [Decoder 2] ───────┘
(128, 64, 64)                                                        (128, 64, 64)     
     │                                                                    │             
     ▼                                                                    ▲             
[Encoder 3] ────────────(Skip Connection)────────────► [Decoder 3] ───────┘
(256, 32, 32)                                          (256, 32, 32)      
     │                                                      │             
     ▼                                                      ▲             
[Encoder 4] ─────(Skip Connection)─────► [Decoder 4] ───────┘
(512, 16, 16)                            (512, 16, 16)      
     │                                        │             
     ▼                                        ▲             
              [   Bottleneck   ] ─────────────┘
              (1024, 8, 8)

```

## 📂 Version History (구현 과정)

| 버전 | Optimizer | 특징 및 개선사항                                                                                   |
|:---:|:---:|:--------------------------------------------------------------------------------------------|
| **v2** | SGD | **초기 프로토타입** U-Net 모델 구축, MSE Loss 사용                                                       |
| **v3** | SGD | **Loss 개선** MSE Loss의 문제점을 파악하고 L1 Loss로 변경                                                 |
| **v4** | **Adam** | **최적화 기법 고도화** `layers_adam.py`를 통해 **Adam Optimizer**를 밑바닥부터 구현. PyTorch와 가장 근접한 수렴 성능 달성. |
| **Torch**| Adam | **Baseline** `nn.Module` 등을 사용한 PyTorch 표준 구현체.                                             |

---
## DataSet (데이터셋)
train_dataset 디렉토리의 dw_data.py 파일을 통해 데이터셋을 다운 받을 수 있습니다.  
다운로드 된 데이터는 ./data 폴더에 저장됩니다.

* 데이터셋 구성
  * LibriSpeech (약 6.3 GB) : 사람의 깨끗한 음성
  * UrbanSound8K (약 5.6 GB) : 공사장 소리, 거리 노랫 소리 등의 노이즈
  * 이 두 가지 소리를 0~15 dB SNR로 mixing하여 학습 데이터셋을 구성
---
## Training (학습)
가장 고도화된 버전인 v4 (Custom Adam) 모델과 PyTorch 모델을 학습시킬 수 있습니다.

```bash
# Custom U-Net (v4) 학습
python train_v4.py

# PyTorch U-Net 학습 (비교군)
python train_torch.py
```   
학습된 가중치는 checkpoints/ 폴더에 .pth 형식으로 자동 저장됩니다.

---
## Inference & Comparison (추론 및 비교)
학습된 모든 버전(v2, v3, v4, Torch)의 모델을 불러와 결과를 시각화하고 오디오 파일(.wav)로 저장합니다.

```Bash
python inference_compare.py
```
---

## 📂 최종 프로젝트 폴더 구조 (Directory Structure)

```plaintext
DeepLearning_Project/
│
├── 📂 checkpoints/           # [저장소] 학습된 모델 가중치 (.pth)
│   ├── v2/                   # - v2 모델 (SGD, Small)
│   ├── v3/                   # - v3 모델 (SGD, Large)
│   ├── v4/                   # - v4 모델 (Adam, Large)
│   └── torch/                # - PyTorch 모델 (Baseline)
│
├── 📂 data/                  # [데이터] 
│   ├── LibriSpeech/          # - Clean Speech 데이터
│   └── noise_datasets/       # - Noise 데이터
│
├── 📂 models/                # [모델] 모델 아키텍처 및 레이어 구현
│   ├── __init__.py          
│   ├── layers.py             # - 기본 레이어 (Conv, Pool, ReLU - v2/v3용)
│   ├── layers_adam.py        # - Adam 최적화 포함 레이어 (v4용)
│   ├── unet_v2.py            # - v2 모델 정의
│   ├── unet_v3.py            # - v3 모델 정의
│   ├── unet_v4.py            # - v4 모델 정의
│   └── unet_torch.py         # - PyTorch 모델 정의
│
├── 📂 mk_dataset/         # [파이프라인] 데이터 전처리 및 로더
│   ├── __init__.py          
│   ├── audio_mixer.py        # - 오디오 믹싱 & SNR 조절 로직
│   ├── spectrogram.py        # - STFT/ISTFT 변환 로직
│   ├── dw_data.py            # - 데이터셋 다운로드 스크립트
│   └── dataset.py            # - PyTorch Dataset 클래스
│
├── utils.py               # [도구] 공통 유틸리티 함수          
│
├── train_v2.py            # v2 학습 스크립트
├── train_v3.py            # v3 학습 스크립트
├── train_v4.py            # v4 학습 스크립트 
├── train_torch.py         # torch 학습 스크립트
│
│
├── inference_v2.py        # v2 추론 스크립트
├── inference_v3.py        # v3 추론 스크립트
├── inference_v4.py        # v4 추론 스크립트 
├── inference_torch.py     # torch 추론 스크립트
├── inference_compare.py   # 모델별 성능 비교 및 시각화 
│
│
├── audio_inference.ipynb  # 2.55초 이상의 오디오 노이즈 제거
├── requirements.txt       # [설정] 필요 라이브러리 목록
└── README.md              # [문서] 프로젝트 설명서
```