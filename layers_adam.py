import torch
from utils import *

# ex) (1, 1, 256, 256), stride = 1, pad = 1
class Conv2d:
    def __init__(self, W, b, stride=1, pad=1):
        # W shape : (FN, C, FH, FW)
        self.W = W  # (16, 1, 3, 3)
        self.b = b  # (16,)
        self.stride = stride
        self.pad = pad

        # for backward
        self.col = None
        self.col_W = None
        self.x_shape = None

        self.t = 0  # 타임 스텝
        
        # Weight에 대한 m, v (모양은 W와 동일)
        self.m_W = torch.zeros_like(self.W)
        self.v_W = torch.zeros_like(self.W)
        
        # Bias에 대한 m, v (모양은 b와 동일)
        self.m_b = torch.zeros_like(self.b)
        self.v_b = torch.zeros_like(self.b)

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        self.x_shape = x.shape  # for backward

        # 출력 크기 계산
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        # im2col 실행
        # col : (N * out_h * out_w, C * FH * FW)
        col = im2col(x, FH, FW, self.stride, self.pad)  # (1*256*256, 1*9)
        self.col = col  # for backward

        # 필터 전개
        # col_W : (FN, C * FH * FW)
        self.col_W = self.W.reshape(FN, -1)  # (16, 9)

        # 행렬 곱 + bias
        # out = x @ W + b
        # (N * out_h * out_w, C * FH * FW) @ (FN, C * FH * FW).T = (N * out_h * out_w, FN)
        # (1*256*256, 9) @ (9, 16) = (1*256*256, 16)
        # (256*256, 16) + (1, 16)
        # torch.matmul = np.dot
        out = torch.matmul(self.col, self.col_W.T) + self.b

        # 출력 형태 복원 (feature map)
        # out : (N * out_h * out_w, FN)
        # -> (N, FN, out_h, out_w)
        out = out.reshape(N, out_h, out_w, -1).permute(0,3,1,2)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # 1. bias 기울기 : 채널별로 합산
        # dout shape : (N, FN, out_h, out_w) -> (N * out_h * out_w, FN)
        # ex) (1, 16, 256, 256) -> (1*256*256, 16)
        dout = dout.permute(0,2,3,1).reshape(-1, FN)
        self.db = torch.sum(dout, dim=0)  # (FN,) = (16,)

        # 2. Weight의 기울기 (행렬 곱)
        # dW = col.T @ dout
        # col : (N * out_h * out_w, C * FH * FW)
        # dout : (N * out_h * out_w, FN)
        # dW : (C * FH * FW, FN)
        # (1*256*256, 9).T @ (1*256*256, 16)
        self.dW = torch.matmul(self.col.T, dout)  # (9, 16)
        self.dW = self.dW.T.reshape(FN, C, FH, FW)  # 원래 필터 모양으로 복구 (16, 1, 3, 3)

        # 3. 입력(x) 데이터 기울기 (dcol -> dx)
        # dcol = dout @ W.T
        # dout : (N * out_h * out_w, FN)
        # col_W: (FN, C * FH * FW)
        # (N * out_h * out_w, FN) @ (FN, C * FH * FW) = (N * out_h * out_w, C * FH * FW)
        # dcol : (N * out_h * out_w, C * FH * FW)
        col_W = self.col_W  # (16, 9)
        dcol = torch.matmul(dout, col_W)  # (256*256, 9)

        # dcol : (N * out_h * out_w, C * FH * FW)
        # -> im2col 출력 shape : (N * out_h * out_w, C * FH * FW)
        # -> col2im : (N, C, H, W)
        dx = col2im(dcol, self.x_shape, FH, FW, self.stride, self.pad)

        return dx

    def step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # SGD
        # self.W -= lr * self.dW
        # self.b -= lr * self.db

        # ----- Adam -----
        self.t += 1

        # 1) 1차 모멘트(m)
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db

        # 2) 2차 모멘트(v)
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW**2)
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db**2)

        # 3) bias correction
        m_W_hat = self.m_W / (1 - beta1**self.t)
        v_W_hat = self.v_W / (1 - beta2**self.t)

        m_b_hat = self.m_b / (1 - beta1**self.t)
        v_b_hat = self.v_b / (1 - beta2**self.t)

        # 4) 파라미터 업데이트
        self.W -= lr * m_W_hat / (torch.sqrt(v_W_hat) + eps)
        self.b -= lr * m_b_hat / (torch.sqrt(v_b_hat) + eps)


class ConvTransposed2d:
    def __init__(self, W, b, stride=2, pad=0, output_pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        self.output_pad = output_pad

        # for backward
        self.x = None
        self.x_flat = None
        self.col_W = None

        self.t = 0  # 타임 스텝

        # Weight에 대한 m, v (모양은 W와 동일)
        self.m_W = torch.zeros_like(self.W)
        self.v_W = torch.zeros_like(self.W)

        # Bias에 대한 m, v (모양은 b와 동일)
        self.m_b = torch.zeros_like(self.b)
        self.v_b = torch.zeros_like(self.b)

    def forward(self, x):
        # x shape : (N, C, H, W)
        N, C, H, W = x.shape
        # W shape : (C, FN, FH, FW)
        # C : 입력 데이터의 채널 수
        # FN : 출력 데이터의 채널 수
        # Conv2d 와 shape이 다른 이유 : ConvTransposed2d는 Conv2d의 역전파 과정과 같다
        C, FN, FH, FW = self.W.shape
        self.x = x  # for backward

        # 출력 크기 계선 (Conv2d의 역연산)
        out_h = (H - 1) * self.stride - 2 * self.pad + FH + self.output_pad
        out_w = (W - 1) * self.stride - 2 * self.pad + FW + self.output_pad
        out_shape = (N, FN, out_h, out_w)

        # 가중치와 입력 행렬 곱 (Conv2d의 backward 연산)
        # W : (C, FN, FH, FW) -> col_W : (C, FN * FH * FW)
        col_W = self.W.reshape(C, -1)

        # x = (N, C, H, W) -> (N * H * W, C)
        x_flat = x.permute(0,2,3,1).reshape(-1, C)
        self.x_flat = x_flat  # for backward

        # 행렬 곱 : out = col_W @ x.T
        # : (N * H * W, FN * FH * FW)
        out = torch.matmul(x_flat, col_W)

        # col2im 통해 이미지 형태로 복원
        # out은 이미 col2im의 입력 형태 : (N * H * W, FN * FH * FW)
        # -> shape : (N, FN ,H ,W)
        out = col2im(out, out_shape, FH, FW, self.stride, self.pad)

        # bias 더하기 (broadcasting)
        out += self.b.reshape(1, FN, 1, 1)

        return out

    def backward(self, dout):
        # out : (N, FN, OH, OW)
        N, FN, OH, OW = dout.shape
        C, FN, FH, FW = self.W.shape
        N, C, H ,W = self.x.shape

        # 1. db
        # (N, FN, OH, OW) -> (FN, N * OH * OW)
        self.db = torch.sum(dout, dim=(0,2,3))

        # 2. dW
        # dout을 im2col로 전개
        # col_dout : (N * OH * OW, FN * FH * FW)
        col_dout = im2col(dout, FH, FW, self.stride, self.pad)

        # dW = x_flat.T * col_dout
        # 헹렬 곱을 위해 reshape
        # x_flat : (N * H * W, C) -> transpose
        # (C, N * H * W) @ (N * OH * OW, FN * FH * FW)
        # = (C, FN * FH * FW)
        dW_flat = torch.matmul(self.x_flat.T, col_dout)
        self.dW = dW_flat.reshape(C, FN, FH, FW)  # 원래 모양으로 복구  

        # 3. dx
        # dx = col_dout @ W.T
        # col_dout : (N * OH * OW, FN * FH * FW)
        # W : (C, FN, FH, FW) -> (FN * FH * FW, C)
        col_W_T = self.W.reshape(C, -1).T

        # (N * OH * OW, FN * FH * FW) @ (FN * FH * FW, C)
        # = (N * OH * OW, C)
        dx_flat = torch.matmul(col_dout, col_W_T)

        # 원래 크기로 복원
        dx = dx_flat.reshape(N, H, W, C).permute(0, 3, 1, 2)

        return dx

    def step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # SGD
        # self.W -= lr * self.dW
        # self.b -= lr * self.db

        # ----- Adam -----
        self.t += 1

        # 1) 1차 모멘트(m)
        self.m_W = beta1 * self.m_W + (1 - beta1) * self.dW
        self.m_b = beta1 * self.m_b + (1 - beta1) * self.db

        # 2) 2차 모멘트(v)
        self.v_W = beta2 * self.v_W + (1 - beta2) * (self.dW**2)
        self.v_b = beta2 * self.v_b + (1 - beta2) * (self.db**2)

        # 3) bias correction
        m_W_hat = self.m_W / (1 - beta1**self.t)
        v_W_hat = self.v_W / (1 - beta2**self.t)

        m_b_hat = self.m_b / (1 - beta1**self.t)
        v_b_hat = self.v_b / (1 - beta2**self.t)

        # 4) 파라미터 업데이트
        self.W -= lr * m_W_hat / (torch.sqrt(v_W_hat) + eps)
        self.b -= lr * m_b_hat / (torch.sqrt(v_b_hat) + eps)


class LeakyReLU:
    def __init__(self, slope=0.01):
        self.slope = slope
        self.mask = None

    def forward(self, x):
        # x가 0이하인 인덱스 저장(True/False)
        self.mask = (x <= 0)
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

    def step(self, lr):
        pass  # 업데이트할 가중치 X


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


class DoubleConv:
    """
    (Conv -> LeakyReLU) * 2 구조를 하나의 레이어
    """

    def __init__(self, in_ch, out_ch, f_size = 3, mid_ch=None):
        if mid_ch is None:
            mid_ch = out_ch

        # 2. He Initialization을 위한 Fan-in 면적 계산
        # 3x3이면 9, 5x5면 25
        self.f_area = f_size * f_size

        # 첫 번째 합성곱 (채널 수 변경: in -> mid)
        self.conv1 = Conv2d(
            he_init(in_ch * self.f_area, (mid_ch, in_ch, f_size, f_size)),
            torch.zeros(mid_ch),
            stride=1,
            pad=1,
        )
        self.act1 = LeakyReLU()

        # 두 번째 합성곱 (채널 수 유지: mid -> out)
        self.conv2 = Conv2d(
            he_init(mid_ch * self.f_area, (out_ch, mid_ch, f_size, f_size)),
            torch.zeros(out_ch),
            stride=1,
            pad=1,
        )
        self.act2 = LeakyReLU()

        # Optimizer용 파라미터 리스트
        self.params = [self.conv1, self.conv2]

    def forward(self, x):
        # x -> Conv1 -> Act1 -> Conv2 -> Act2
        out = self.conv1.forward(x)
        out = self.act1.forward(out)
        out = self.conv2.forward(out)
        out = self.act2.forward(out)
        return out

    def backward(self, dout):
        # 역순으로 전파
        # Act2 -> Conv2 -> Act1 -> Conv1
        dout = self.act2.backward(dout)
        dout = self.conv2.backward(dout)
        dout = self.act1.backward(dout)
        dout = self.conv1.backward(dout)
        return dout

    def step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # 내부 컨볼루션 레이어들의 가중치 업데이트
        self.conv1.step(lr, beta1, beta2, eps)
        self.conv2.step(lr, beta1, beta2, eps)


# sigmoid
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        # y = 1 / (1 + e^-x)
        self.out = 1 / (1 + torch.exp(-x))
        return self.out
    
    def backward(self, dout):
        # sigmoid 미분 : y * (1 - y)
        dx = dout * self.out * (1.0 - self.out)
        return dx


# 손실함수 (MSE)
class MSELoss:
    def __init__(self):
        self.diff = None
        self.N = None

    def forward(self, y, t):
        self.diff = y - t
        self.N = y.numel()  # 전체 요소 개수
        loss = torch.sum(self.diff ** 2)/self.N

        return loss
    
    def backward(self):
        # MSE 미분 : 2(y - t)/N
        dout = 2 * self.diff / self.N
        
        return dout 


# 손실함수 (L1 Loss)
# L1 = |y - t| / N
class L1Loss:
    def __init__(self):
        self.diff = None
        self.N = None

    def forward(self, y, t):
        self.diff = y - t
        self.N = y.numel()
        loss = torch.sum(torch.abs(self.diff)) / self.N

        return loss

    def backward(self):
        # 미분 : (1/N) * sign(pred - target)
        grad_direction = torch.sign(self.diff)
        dout = grad_direction / self.N

        return dout
