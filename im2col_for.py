import torch
import torch.nn.functional as F


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Args:
        input_data: (N, C, H, W) Tensor (GPU or CPU)
        filter_h, filter_w: 필터 크기
    Returns:
        (N * out_h * out_w, C * filter_h * filter_w) 2D Tensor
    """
    N, C, H, W = input_data.shape

    # 1. 출력 크기 계산
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 2. 패딩 적용 (PyTorch의 F.pad는 (Left, Right, Top, Bottom) 순서)
    # GPU에 있는 텐서를 그대로 패딩하므로, 결과(img)도 GPU에 유지됩니다.
    img = F.pad(input_data, (pad, pad, pad, pad), mode="constant", value=0)

    # 3. 거대 텐서를 위한 공간 생성 (가장 중요!)
    # 입력 데이터가 있는 장소(device)와 같은 곳에 빈 텐서를 만듭니다.
    col = torch.zeros(
        (N, C, filter_h, filter_w, out_h, out_w),
        dtype=input_data.dtype,
        device=input_data.device,
    )

    # 4. 데이터 채워넣기 (Slicing on GPU)
    # Python Loop는 CPU가 돌지만, 내부의 슬라이싱과 복사 연산은 GPU가 수행합니다.
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w

            # GPU VRAM 내부에서의 고속 복사 (Device Transfer 없음)
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    # 5. 차원 순서 변경 (NumPy transpose -> PyTorch permute)
    # (N, C, fh, fw, oh, ow) -> (N, oh, ow, C, fh, fw)
    col = col.permute(0, 4, 5, 1, 2, 3).contiguous()

    # 6. 2차원으로 펼치기
    col = col.view(N * out_h * out_w, -1)

    return col
