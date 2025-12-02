딥러닝 모델 학습 및 추론을 위한 핵심 파이썬 모듈들이 위치합니다.

## 1. 모듈 구성
### `models/`
- 모델 아키텍처를 정의합니다.
- `parts.py`: DoubleConv, Down, Up 등 U-Net의 구성 요소 블록
- `unet.py`: 전체 U-Net 모델 조립 (`parts.py` 활용)

### 'dataloader.py'

### `trainer/`
- 실제 학습 루프(Loop)를 관리합니다.
- `trainer.py`: Epoch 반복, Loss 계산, Backpropagation, Optimizer 업데이트 로직 포함.

### `utils/`
- 보조 기능을 담당합니다.
- `logger.py`: 학습 진행 상황 로깅
- `metrics.py`: PSNR, SSIM 등 평가 지표 계산 함수
- `checkpoint.py`: 모델 가중치 저장 및 로드

## 2. 사용법
이 폴더의 모듈들은 루트 디렉토리의 `main.py`나 `train.py`에서 호출되어 사용됩니다.