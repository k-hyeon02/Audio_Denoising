import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    입력 데이터를 필터가 적용될 영역마다 잘라서 열(column)로 전개하는 함수 (PyTorch Ver.)
    
    Args:
        input_data (torch.Tensor): (batch_size, n_channel, h, w)
        filter_h, filter_w : 필터 높이, 너비
    
    Returns:
        col (torch.Tensor): (윈도우 총 개수, 윈도우 하나의 크기)
    """
    
    N, C, H, W = input_data.shape

    # 1. 출력 크기 계산
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 2. 패딩 적용 (PyTorch의 F.pad는 (Left, Right, Top, Bottom) 순서)
    # input_data가 4차원이므로 마지막 2개 차원(W, H)에 대해 패딩 적용
    img = F.pad(input_data, (pad, pad, pad, pad), mode='constant', value=0)

    # 3. 거대 행렬을 위한 공간 생성
    # 입력 데이터와 같은 장치(CPU/GPU)와 같은 타입으로 생성
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype=input_data.dtype, device=input_data.device)

    # 4. 데이터 채워넣기 (슬라이딩 윈도우 영역을 복사)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w

            # 이미지에서 슬라이싱하여 할당
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    # 5. 모양 바꾸기 (permute & reshape)
    # (N, C, filter_h, filter_w, out_h, out_w)
    # -> (N, out_h, out_w, C, filter_h, filter_w) 로 축 변경
    col = col.permute(0, 4, 5, 1, 2, 3).contiguous()

    # 최종적으로 2차원 행렬로 펼친다.
    # 행: 윈도우 하나 (이미지 조각) / 열: 픽셀 값들
    col = col.view(N * out_h * out_w, -1)
    
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    2차원 배열을 다시 원래 4차원 이미지 형상으로 복원한다 (PyTorch Ver.)
    주로 역전파(Backward)시 기울기를 원래 이미지 형태로 돌릴 때 사용
    
    Args:
        col (torch.Tensor): (N*OH*OW, C*FH*FW)
        input_shape: 튜플 (N, C, H, W)
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 1. 모양 복구
    # (N*OH*OW, -1) -> (N, OH, OW, C, FH, FW) -> (N, C, FH, FW, OH, OW)
    col = col.view(N, out_h, out_w, C, filter_h, filter_w)
    col = col.permute(0, 3, 4, 5, 1, 2).contiguous()

    # 2. 빈 이미지 생성 (패딩된 크기만큼 생성)
    # 기존 코드 로직 유지: H + 2*pad + stride - 1
    img_h = H + 2*pad + stride - 1
    img_w = W + 2*pad + stride - 1
    img = torch.zeros((N, C, img_h, img_w), dtype=col.dtype, device=col.device)
    
    # 3. 픽셀 누적 (Gradient Accumulation)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            
            # += 연산으로 겹치는 부분의 값을 더해줌 (역전파 핵심)
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # 4. 패딩 제거하고 원래 크기로 자름
    return img[:, :, pad:H + pad, pad:W + pad]