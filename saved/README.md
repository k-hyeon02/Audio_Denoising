# 📂 Saved Output Directory

학습 과정에서 생성된 모델 가중치(Weights)와 로그(Logs)가 자동으로 저장되는 곳입니다.
**이 폴더의 내용은 실험 결과물이므로 Git에 포함되지 않습니다.**

## 1. 폴더 구조
학습이 시작되면 `실험명_날짜시간`(예: `unet_v1_20231025`) 폴더가 자동 생성됩니다.

```text
saved/
└── unet_v1_20231025_120000/
    ├── checkpoints/
    │   ├── best_model.pth  # 검증 손실이 가장 낮은 모델
    │   └── last_model.pth  # 학습이 끝난 시점의 모델
    └── logs/
        ├── events.out.tfevents... # Tensorboard 로그 파일
        └── train.log              # 텍스트 로그 파일

