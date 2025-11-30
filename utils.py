import torch


def im2col(x, kernel_h, kernel_w, stride=1, padding=1):
    N, C, H, W = x.shape

    # 1. 패딩 및 출력 크기 계산
    x_padded = torch.nn.functional.pad(
        x, (padding, padding, padding, padding), mode="constant", value=0
    )
    out_h = (H + 2 * padding - kernel_h) // stride + 1
    out_w = (W + 2 * padding - kernel_w) // stride + 1

    # 2. 커널 내부(Local) 좌표: (offset_h, offset_w)
    # offset_h (세로/행): 커널 내에서 천천히 변함
    # [0, 0, 0, 1, 1, 1, 2, 2, 2]
    offset_h = torch.arange(kernel_h).reshape(-1, 1).repeat(1, kernel_w).reshape(-1)

    # offset_w (가로/열): 커널 내에서 빠르게 변함
    # [0, 1, 2, 0, 1, 2, 0, 1, 2]
    offset_w = torch.arange(kernel_w).reshape(1, -1).repeat(kernel_h, 1).reshape(-1)

    # 채널 (k)
    offset_c = torch.arange(C).reshape(-1, 1).repeat(1, kernel_h * kernel_w).reshape(-1)

    # 3. 윈도우 시작점(Global) 좌표: (start_h, start_w)

    # start_h (세로 시작점):
    # 가로로 훑는 동안(Inner Loop)은 h좌표가 고정되어야 함 -> [0, 0, ..., 0, stride, stride, ...]
    # 형태: (out_h, 1) -> 가로로 out_w 만큼 복사
    start_h = torch.arange(out_h) * stride
    start_h = start_h.reshape(-1, 1).repeat(1, out_w).reshape(-1)

    # start_w (가로 시작점):
    # 가로로 훑는 동안 계속 변해야 함 -> [0, stride, 2*stride, ...]
    # 형태: (1, out_w) -> 세로로 out_h 만큼 복사
    start_w = torch.arange(out_w) * stride
    start_w = start_w.reshape(1, -1).repeat(out_h, 1).reshape(-1)

    # 4. 좌표 결합 (Broadcasting)
    # 전체 행 좌표 = 커널 내부 h + 윈도우 시작 h
    h_idx = offset_h.reshape(-1, 1) + start_h.reshape(1, -1)

    # 전체 열 좌표 = 커널 내부 w + 윈도우 시작 w
    w_idx = offset_w.reshape(-1, 1) + start_w.reshape(1, -1)

    c_idx = offset_c.reshape(-1, 1).repeat(1, out_h * out_w)

    # 5. 데이터 추출 cols[b, c, h, w] 순서
    cols = x_padded[:, c_idx, h_idx, w_idx]

    return cols
