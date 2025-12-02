학습 실험을 제어하는 하이퍼파라미터 및 환경 설정 파일들을 관리합니다.
코드를 수정하지 않고 `.yaml` 파일만 수정하여 다양한 실험을 수행할 수 있습니다.

## 1. 파일 목록
- **`config.yaml`**: 기본 설정 파일 (Base Configuration)
- **`experiments/`**: 다양한 실험을 위한 변형 설정 파일들

## 2. 주요 파라미터 설명 (`config.yaml`)

### Data
- `data_dir`: 데이터셋 루트 경로
- `img_size`: 입력 이미지 크기 (예: 256)
- `batch_size`: 한 번에 학습할 이미지 수 (GPU 메모리에 맞춰 조절)

### Model
- `architecture`: 사용할 모델 이름 (예: `Unet`)
- `in_channels`: 입력 채널 수 (RGB=3, Grayscale=1)

### Train
- `epochs`: 전체 데이터셋 반복 횟수
- `lr`: Learning Rate (학습률, 예: 1e-4)
- `loss`: 손실 함수 (예: `MSE`, `L1`)
- `seed`: 재현성을 위한 Random Seed 값