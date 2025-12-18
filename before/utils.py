import torch
import numpy as np
import os

def im2col(x, filter_h, filter_w, stride=1, pad=1):
    N, C, H, W = x.shape

    # 1. 패딩 및 출력 크기 계산
    img = torch.nn.functional.pad(
        x, (pad, pad, pad, pad), mode="constant", value=0
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
    # 형태: (out_h, 1) -> 가로로 out_w 만큼 복사
    # shape : (out_h * out_w,)
    start_h = torch.arange(out_h) * stride
    start_h = start_h.reshape(-1, 1).repeat(1, out_w).reshape(-1)  

    # start_w (가로 시작점):
    # 가로로 훑는 동안 계속 변해야 함 -> [0, stride, 2*stride, ...]
    # 형태: (1, out_w) -> 세로로 out_h 만큼 복사
    # shape : (out_h * out_w,)
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

    # reshape :
    # (N, C * filter_h * filter_w, out_h * out_w)
    # -> (N * out_h * out_w, C * filter_h * filter_w)
    col = col.permute(0, 2, 1).reshape(N * out_h * out_w, -1)

    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=1):
    """
    col: (N * out_h * out_w, C * filter_h * filter_w)
    input_shape: (N, C, H, W) - 복원할 입력 이미지의 크기
    """
    N, C, H, W = input_shape

    # 1. 출력 높이/너비 계산
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    # 2. 텐서 모양 복구 (Reshape)
    # 입력된 평평한 col를 (N, C, filter_h, filtier_w, out_h, oout_w) 형태로 다시 복원
    # im2col 출력 형태를 고려하여 reshape
    col = col.reshape(N, out_h * out_w, -1).permute(0, 2, 1)
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


# 가중치 초기화 함수 (He Initialization)
def he_init(fan_in, shape):
    return torch.randn(shape) * np.sqrt(2.0 / fan_in)

# device 설정
def get_device():
    """장치 자동 감지: Mac(MPS), NVIDIA(CUDA), CPU 순서"""
    if torch.backends.mps.is_available():
        print("mps available")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("cuda available")
        return torch.device("cuda")
    else:
        print("only cpu available")
        return torch.device("cpu")


# 모델 gpu 이동 및 저장     / model.to(device) 사용
# def move_layer_to_device(layer, device):
#     """레이어 내부의 W, b를 찾아 디바이스로 이동"""
#     # 1. 기본 레이어 (Conv2d, ConvTransposed2d 등)
#     if hasattr(layer, "W") and layer.W is not None:
#         layer.W = layer.W.to(device)
#     if hasattr(layer, "b") and layer.b is not None:
#         layer.b = layer.b.to(device)

#     # 2. 중첩 레이어 (DoubleConv)
#     if hasattr(layer, "params"):
#         for sub_layer in layer.params:
#             move_layer_to_device(sub_layer, device)


# def move_model_to_device(model, device):
#     """UNet 전체 파라미터 이동"""
#     print(f"Moving model to {device}...")
#     for module in model.modules():
#         move_layer_to_device(module, device)


# 가중치 저장
def save_checkpoint(model, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved to: {save_path}")



# 저장한 가중치 불러오기
def load_checkpoint(model, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from: {load_path}")
    return model



# PSNR 계산 : PSNR = 10 * log10(MAX^2 / MSE)
def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    if mse == 0: # 둘이 완전히 똑같으면 무한대
        return 100.0
    
    # 데이터가 0~1 혹은 -1~1로 정규화되어 있다고 가정할 때 MAX=1.0으로 잡는 것이 일반적입니다.
    max_val = 1.0 
    psnr = 10 * torch.log10((max_val**2) / mse)
    return psnr
