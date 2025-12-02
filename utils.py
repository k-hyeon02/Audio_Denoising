import torch


def im2col(x, filter_h, filter_w, stride=1, pad=1):
    N, C, H, W = x.shape

<<<<<<< HEAD
    # 1. 패딩 및 출력 크기 계산 
    x_padded = torch.nn.functional.pad(
        x, (padding, padding, padding, padding), mode="constant", value=0
=======
    # 1. 패딩 및 출력 크기 계산
    img = torch.nn.functional.pad(
        x, (pad, pad, pad, pad), mode="constant", value=0
>>>>>>> ec24e95a (append col2im)
    )
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 2. 필터 내부(Local) 좌표: (offset_h, offset_w)
    # offset_h (세로/행): 필터 내에서 천천히 변함
    # [0, 0, 0, 1, 1, 1, 2, 2, 2]
    # shape : (C * filter_h * filter_w,)
    offset_h = torch.arange(filter_h).reshape(-1, 1).repeat(1, filter_w).reshape(-1).repeat(C)

    # offset_w (가로/열): 필터 내에서 빠르게 변함
    # [0, 1, 2, 0, 1, 2, 0, 1, 2]
    offset_w = torch.arange(filter_w).reshape(1, -1).repeat(filter_h, 1).reshape(-1).repeat(C)

    # 채널 (k) : (C * filter_h * filter_w,)
    offset_c = torch.arange(C).reshape(-1, 1).repeat(1, filter_h * filter_w).reshape(-1)

    # 3. 윈도우 시작점(Global) 좌표: (start_h, start_w)

    # start_h (세로 시작점):
    # 가로로 훑는 동안(Inner Loop)은 h좌표가 고정되어야 함 -> [0, 0, ..., 0, stride, stride, ...]
<<<<<<< HEAD
    # 형태: (out_h, 1) -> 가로(out_w)로 복사
=======
    # 형태: (out_h, 1) -> 가로로 out_w 만큼 복사
    # shape : (out_h * out_w,)
>>>>>>> ec24e95a (append col2im)
    start_h = torch.arange(out_h) * stride
    start_h = start_h.reshape(-1, 1).repeat(1, out_w).reshape(-1)  

    # start_w (가로 시작점):
    # 가로로 훑는 동안 계속 변해야 함 -> [0, stride, 2*stride, ...]
<<<<<<< HEAD
    # 형태: (1, out_w) -> 세로(out_h)로 복사
=======
    # 형태: (1, out_w) -> 세로로 out_h 만큼 복사
    # shape : (out_h * out_w,)
>>>>>>> ec24e95a (append col2im)
    start_w = torch.arange(out_w) * stride
    start_w = start_w.reshape(1, -1).repeat(out_h, 1).reshape(-1) 

    # 4. 좌표 결합 (Broadcasting)
    # 전체 행 좌표 = 필터 내부 h + 윈도우 시작 h
    # (C * filter_h * filter_w,) + (1, out_h * out_w) = (C * filter_h * filter_w, out_h * out_w)
    h_idx = offset_h.reshape(-1, 1) + start_h.reshape(1, -1)

    # 전체 열 좌표 = 필터 내부 w + 윈도우 시작 w
    w_idx = offset_w.reshape(-1, 1) + start_w.reshape(1, -1)

    c_idx = offset_c.reshape(-1, 1).repeat(1, out_h * out_w)

    # 5. 데이터 추출 col[b, c, h, w] 순서
    # shape = (N, C * filter_h * filter_w, out_h * out_w)
    col = img[:, c_idx, h_idx, w_idx]

<<<<<<< HEAD
    return cols

=======
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=1):
    """
    col: (N, C * filter_h * filter_w, out_h * out_w)
    input_shape: (N, C, H, W) - 복원할 입력 이미지의 크기
    """
    N, C, H, W = input_shape

    # 1. 출력 높이/너비 계산
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 2. 텐서 모양 복구 (Reshape)
    # 입력된 평평한 col를 (N, C, filter_h, filtier_w, out_h, oout_w) 형태로 다시 복원
    # im2col 출력 형태를 고려하여 reshape
    col = col.reshape(N, C, filter_h, filter_w, out_h, out_w)

    # 3. 빈 캔버스 생성 (패딩이 포함된 크기)
    img = torch.zeros((N, C, H + 2 * pad, W + 2 * pad), 
                           dtype=col.dtype, device=col.device)

    # 4. 텐서 슬라이싱 & 덧셈 (Matrix Operation)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad : H+pad, pad : W+pad]
>>>>>>> ec24e95a (append col2im)
