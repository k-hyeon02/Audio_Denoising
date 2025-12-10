import torch
import torch.nn.functional as F
import components.imcol as imcol
from components.tools import *

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = 1, device=None):
        super().__init__()
        # 가중치 초기화 using He Initialization
        scale = torch.sqrt(torch.tensor(2.0 / (in_channels * kernel_size ** 2)))
        self.params['W'] = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device=device) * scale # (FN, C, FH, FW)

        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        Args:
            x (N, C, H, W)
        
        Returns:
            out (N, FN, OH, OW)
        """
        self.cache = x  # 역전파용 저장
        W = self.params['W']

        N, C, H, W_in = x.shape
        FN, C, FH, FW = W.shape

        # 1. 출력 크기 계산
        OH = (H + 2*self.pad - FH) // self.stride + 1
        OW = (W_in + 2*self.pad - FW) // self.stride + 1   

        out = torch.zeros((N, FN, OH, OW), device=x.device)

        # 2. image to column
        col = imcol.im2col(x, FH, FW, self.stride, self.pad)  # (N*OH*OW, C*FH*FW)

        # 3. filter to 2D
        # (FN, C, FH, FW) -> (FN, C*FH*FW) -> (C*FH*FW, FN)
        col_W = W.reshape(FN, -1).T    # shape : (필터 크기, 필터 개수)

        # <--- 디버깅 코드 추가 시작 --->
        print(f"DEBUG: Conv2d Input (col) Device: {col.device}")
        W = self.params["W"]
        col_W = W.reshape(FN, -1).T
        print(f"DEBUG: Conv2d Weight (col_W) Device: {col_W.device}")
        # <--- 디버깅 코드 추가 끝 --->

        # 4. (데이터 개수, 필터 크기) @ (필터 크기, 필터 개수) = (데이터 개수, 필터 개수) = (N*OH*OW, FN)
        out = col @ col_W

        # 5. Reshape to 4D img
        out = out.reshape(N, OH, OW, -1).permute(0, 3, 1, 2).contiguous()    # (N, FN, OH, OW) 채널이 FN으로 바뀐다.

        self.cache = (x, col, col_W)

        return out

    def backward(self, dout):
        """
        Args:
            dout (N, FN, OH, OW)
        
        Returns:
            dx (N, C, H, W)
        """
        W = self.params['W']
        x, col, col_W = self.cache
        FN, C, FH, FW = W.shape

        # dout을 행렬곱을 위해 펼친다.
        # (F, FN, OH, OW) -> (N, OH, OW, FN) -> (N*OH*OW, FN)
        dout =  dout.permute(0, 2, 3, 1).contiguous().reshape(-1, FN)

        # dW = x.T @ dout
        dW = col.T @ dout
        # (C*FH*FW, FN) -> (FN, C, FH, FW)
        self.grads['W'] = dW.permute(1, 0).contiguous().reshape(FN, C, FH, FW)

        # dcol = dout @ W.T
        dcol = dout @ col_W.T

        # col2im으로 이미지 크기 복원
        dx = imcol.col2im(dcol, x.shape, FH, FW, self.stride, self.pad)

        return dx

# H = 32
# stride=2
# pad=1
# FH=4
# out_h = (H - 1) * stride - 2 * pad + FH
# print(out_h)
class ConvTranspose2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 4, stride = 2, pad = 1, device=None):
        super().__init__()
        # 가중치 초기화 using He Initialization
        scale = torch.sqrt(torch.tensor(2.0 / (in_channels * kernel_size ** 2)))
        self.params['W'] = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device=device) * scale # (C, FN, FH, FW)

        self.stride = stride
        self.pad = pad

    def forward(self, x):
        """
        [Forward 원리]
        입력 x를 펴서(flatten), 가중치와 곱한 뒤,
        col2im을 사용해 겹치는 부분을 더해가며 큰 이미지를 만듭니다.
        """
        W_param = self.params['W']
        N, C, H, W = x.shape
        C, FN, FH, FW = W_param.shape
        
        # 1. 출력 크기 계산
        out_h = (H - 1) * self.stride - 2 * self.pad + FH
        out_w = (W - 1) * self.stride - 2 * self.pad + FW
        
        out_shape = (N, FN, out_h, out_w)
        # 2. 입력 x 변형 (행렬 곱을 위해)
        # (N, C, H, W) -> (N, H, W, C) -> (N*H*W, C)
        # 여기서 C는 in_channels입니다.
        x_flat = x.permute(0, 2, 3, 1).contiguous().reshape(-1, C)
        
        # 3. 가중치 변형
        # (In, Out, KH, KW) -> (In, Out*KH*KW)
        col_W = W_param.reshape(C, -1)
        
        # 4. 행렬 곱 (Convolution의 역연산)
        # (N*H*W, C) @ (C, FN*FH*FW) -> (N*H*W, FN*FH*FW)
        out = x_flat @ col_W
        
        # 5. col2im으로 이미지 복원 (작은 이미지 -> 큰 이미지)
        # out_col을 다시 (N, Out, OH, OW) 형태로 조립합니다.
        out = imcol.col2im(out, out_shape, FH, FW, self.stride, self.pad)
        
        # 역전파를 위해 저장
        self.cache = (x, x_flat, col_W)
        
        return out

    def backward(self, dout):
        W_param = self.params['W']
      
        C, FN, FH, FW = W_param.shape
        x, x_flat, col_W = self.cache
        N, C, H ,W = x.shape

        # 2. dW
        # dout을 im2col로 전개
        # col_dout : (N * OH * OW, FN * FH * FW)
        col_dout = imcol.im2col(dout, FH, FW, self.stride, self.pad)

        # dW = x_flat.T * col_dout
        # 헹렬 곱을 위해 reshape
        # x_flat : (N * H * W, C) -> transpose
        # (C, N * H * W) @ (N * OH * OW, FN * FH * FW)
        # = (C, FN * FH * FW)
        dW_flat = x_flat.T @ col_dout
        self.dW = dW_flat.reshape(C, FN, FH, FW)  # 원래 모양으로 복구  

        # 3. dx
        # dx = col_dout @ W.T
        # col_dout : (N * OH * OW, FN * FH * FW)
        # W : (C, FN, FH, FW) -> (FN * FH * FW, C)
        col_W_T = W_param.reshape(C, -1).T

        # (N * OH * OW, FN * FH * FW) @ (FN * FH * FW, C)
        # = (N * OH * OW, C)
        dx_flat = torch.matmul(col_dout, col_W_T)

        # 원래 크기로 복원
        dx = dx_flat.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return dx
if __name__ == "__main__":
    t1 = torch.randn(3, 64, 32, 32)
    t2 = torch.randn(3, 32, 64, 64)
    test = ConvTranspose2d(64, 32, kernel_size=4)
    res1 = test.forward(t1)
    res2 = test.backward(t2)
    print(res1.shape)
    print(res2.shape)

# Activation

class sigmoid(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 1 / (1 + torch.exp(x))
    
    def backward(self, dout):
        
        return 

class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)    # 0 이하인 인덱스 저장
        out = x.clone()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class LeakyReLU:
    def __init__(self, slope=0.01):
        self.slope = slope
        self.mask = None
    def forward(self, x):
        # x가 0이하인 인덱스 저장(True/False)
        self.mask = x <= 0
        out = x.clone()
        out[self.mask] *= self.slope

        return out

    def backward(self, dout):
        # 상류층에서 흘러온 기울기 복사
        # 양수 구간은 미분값 1
        # dx = dout * 1
        dx = dout.clone()

        # 음수 구간은 slope(0.01) 곱해줌
        dx[self.mask] *= self.slope

        return dx


# skip connection

class Concat:
    """
    Skip Connection을 위해 채널 방향(dim=1)으로 데이터를 합치고,
    역전파 시 기울기를 분배하는 레이어
    """
    def __init__(self):
        self.split_idx = None

    def forward(self, x1, x2):
        # x1 : Decoder에서 올라온 데이터
        # x2 : Encoder에서 가져온 skip connection 데이터
        # x1 + x2
        # shape : (N, C1, H, W), (N, C2, H, W)

        self.split_idx = x1.shape[1]  # x1의 채널 수 저장
        out = torch.cat([x1, x2], dim=1)  # 채널 방향으로 연결 

        return out

    def backward(self, dout):
        # dout : (N, C1+C2, H, W)
        # 합쳐진 순서대로 다시 자름
        # dx1 : Decoder 쪽으로 내려갈 기울기
        # dx2 : Encoder(skip connection) 쪽으로 넘어갈 기울기
        dx1 = dout[:, :self.split_idx, :, :]  # ~ split_idx 앞까지 자름
        dx2 = dout[:, self.split_idx:, :, :]  # split_idx ~ 부터

        return dx1, dx2


# 손실함수 (MSE)
class MSELoss:
    def __init__(self):
        self.diff = None
        self.N = None

    def forward(self, y, t):
        self.diff = y - t
        self.N = y.numel()  # 전체 요소 개수
        loss = torch.sum(self.diff**2) / self.N

        return loss

    def backward(self):
        # MSE 미분 : 2(y - t)/N
        dout = 2 * self.diff / self.N

        return dout


# Batch Normal

# class BatchNorm(Module):        # 사용 x
#     def __init__(self, num_features, eps=1e-5, momentum=0.9):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.momentum = momentum

#         # 학습 가능한 파라미터
#         self.params['gamma'] = np.ones((num_features, 1))      # scale
#         self.params['beta'] = np.zeros((num_features, 1))      # shift

#         # gradient 저장 공간
#         self.grads['gamma'] = np.zeros_like(self.params['gamma'])
#         self.grads['beta'] = np.zeros_like(self.params['beta'])

#         # moving average (평가모드에서 사용)
#         self.moving_mean = np.zeros((num_features, 1))
#         self.moving_var = np.ones((num_features, 1))

#     def forward(self, x):
#         # x shape: (batch_size, num_features) 또는 (batch_size, num_features, height, width)
#         # 여기서는 2D 입력 가정 (batch_size, num_features)
#         self.cache = x.copy()
#         batch_size = x.shape[0]

#         if self._train_mode:
#             # 학습 모드: 배치 통계 사용
#             mu = np.mean(x, axis=0, keepdims=True)
#             var = np.var(x, axis=0, keepdims=True)

#             # moving average 업데이트
#             self.moving_mean = (self.momentum * self.moving_mean +
#                               (1 - self.momentum) * mu)
#             self.moving_var = (self.momentum * self.moving_var +
#                              (1 - self.momentum) * var)

#             # 정규화
#             x_norm = (x - mu) / np.sqrt(var + self.eps)
#             out = self.params['gamma'] * x_norm + self.params['beta']

#             # backward에서 사용하기 위해 cache 저장
#             self.cache = (x, mu, var, x_norm)

#         else:
#             # 평가 모드: moving average 사용
#             x_norm = (x - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
#             out = self.params['gamma'] * x_norm + self.params['beta']

#         return out

#     def backward(self, dout):
#         # dout shape: 입력과 같은 shape
#         x, mu, var, x_norm = self.cache

#         # gamma, beta gradient 계산
#         batch_size = x.shape[0]
#         self.grads['gamma'] = np.sum(dout * x_norm, axis=0, keepdims=True)
#         self.grads['beta'] = np.sum(dout, axis=0, keepdims=True)

#         # 입력 x에 대한 gradient 계산
#         dx_norm = dout * self.params['gamma']
#         dvar = np.sum(dx_norm * (x - mu) * -0.5 * (var + self.eps) ** -1.5, axis=0, keepdims=True)
#         dmu = np.sum(dx_norm * -1 / np.sqrt(var + self.eps), axis=0, keepdims=True) + \
#               np.sum(dvar * -2 * (x - mu) / batch_size, axis=0, keepdims=True)
#         dx = dx_norm / np.sqrt(var + self.eps) + \
#              2 * dvar * (x - mu) / batch_size + \
#              dmu / batch_size

#         return dx


# # --- 테스트 ---
# if __name__ == "__main__":
#     # 1. 입력 데이터 (배치 1, 채널 3, 16x16 이미지)
#     x = np.random.randn(1, 3, 16, 16)

#     # 2. Transpose Conv 레이어 생성 (채널 3->16, 2배 확대)
#     # Stride=2를 줘야 2배로 커집니다.
#     t_conv = ConvTranspose2d(in_channels=3, out_channels=16, kernel_size=2, stride=2, pad=0)

#     # 3. 순전파
#     out = t_conv.forward(x)
#     print(f"Input Shape:  {x.shape}")
#     print(f"Output Shape: {out.shape}")

#     # 예상 결과: 16x16 -> 32x32
#     # Output H = (16-1)*2 - 0 + 2 = 32

#     # 4. 역전파
#     dout = np.random.randn(*out.shape)
#     dx = t_conv.backward(dout)
#     print(f"Grad Input(dx) Shape: {dx.shape}")
#     print(f"Grad Weight(dW) Shape: {t_conv.grads['W'].shape}")
