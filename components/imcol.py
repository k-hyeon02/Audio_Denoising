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
    
    col = F.unfold(
        input_data,
        kernel_size=(filter_h, filter_w),
        stride=stride,
        padding = pad
    )       # (N, C*FH)*FW, OH*OW)

    
    # 5. 모양 바꾸기 (permute & reshape)
    # (N, C, filter_h, filter_w, out_h, out_w)
    # -> (N, out_h, out_w, C, filter_h, filter_w) 로 축 변경
    col = col.permute(0, 2, 1).contiguous()

    # 최종적으로 2차원 행렬로 펼친다.
    # 행: 윈도우 하나 (이미지 조각) / 열: 픽셀 값들
    col = col.view(-1, C*filter_h*filter_w)
    
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
    # (N*OH*OW, C*FH*FW) → (N, C*FH*FW, OH*OW)
    col = col.view(N, out_h * out_w, -1)
    col = col.permute(0, 2, 1).contiguous()

    img = F.fold(
        col,
        output_size=(H, W),
        kernel_size=(filter_h, filter_w),
        stride=stride,
        padding=pad
    )
    

    # 4. 패딩 제거하고 원래 크기로 자름
    return img