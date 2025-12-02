import torch
from core.utils import im2col, col2im

# ex) (1, 1, 256, 256), stride = 1, pad = 1
class Conv2d:
    def __init__(self, W, b, stride=1, pad=1):
        self.W = W  # (16, 1, 3, 3)
        self.b = b  # (16,)
        self.stride = stride
        self.pad = pad

        # for backward
        self.col = None
        self.col_W = None
        self.x_shape = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        self.x_shape = x.shape  # for backward

        # 출력 크기 계산
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        # im2col 실행
        # col : (N, C * FH * FW, out_h * out_w)
        col = im2col(x, FH, FW, self.stride, self.pad)  # (1, 1*9, 256*256)

        # (N, C * FH * FW, out_h * out_w) -> (C * FH * FW, N * out_h * out_w) : 행렬 연산 위해 재배치
        # (1, 1*9, 256*256) -> (1*9, 1*256*256)
        col = col.transpose(0,1).reshape(C * FH * FW, -1)
        self.col = col  # for backward

        # 필터 전개
        self.col_W = self.W.reshape(FN, -1)  # (16, 9)

        # 행렬 곱 + bias
        # (FN, C * FH * FW) @ (C * FH * FW, N * out_h * out_w) = (FN, N * out_h * out_w)
        # (16, 9) * (9, 1*256*256) = (16, 1*256*256)
        # (16, 256*256) + (16, 1)
        # torch.matmul = np.dot
        out = torch.matmul(self.col_W, self.col) + self.b.reshape(-1, 1)

        # 출력 형태 복원 (feature map)
        # out : (FN, N * out_h * out_w)
        # -> (N, FN, out_h, out_w)
        out = out.reshape(FN, N, out_h, out_w).transpose(0,1)

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape

        # 1. bias 기울기 : 채널별로 합산
        # dout shape : (N, FN, out_h, out_w) -> (FN, N * out_h * out_w)
        # ex) (1, 16, 256, 256) -> (16, 1*256*256)
        dout = dout.transpose(0,1).reshape(FN, -1)
        self.db = torch.sum(dout, dim=1)  # (FN,) = (16,)

        # 2. Weight의 기울기 (행렬 곱)
        # dW = dout * col.T
        # dout : (FN, N * out_h * out_w)
        # col : (C * FH * FW, N * out_h * out_w)
        # dW : (FN, C * FH * FW)
        # (16, 1*256*256) * (1*9, 1*256*256)T
        self.dW = torch.matmul(dout, self.col.transpose(0, 1))  # (16, 9)
        self.dW = self.dW.reshape(FN, C, FH, FW)  # 원래 필터 모양으로 복구 (16, 1, 3, 3)

        # 3. 입력(x) 데이터 기울기 (dcol -> dx)
        # dcol = W.T * dout
        # W(flat) : (FN, -1) -> transpose -> (-1, FN)
        # (C * FH * FW, FN) @ (FN, N * out_h * out_w) = (C * FH * FW, N * out_h * out_w)
        # dcol : (C * FH * FW, N * out_h * out_w)
        col_W = self.W.reshape(FN, -1)  # (9, 16)
        dcol = torch.matmul(col_W.T, dout)  # (9, 256*256)

        # dcol : (C * FH * FW, N * out_h * out_w)
        # -> im2col 출력 shape : (N, C * FH * FW, out_h * out_w)
        # -> col2im : (N, C, H, W)
        dcol = dcol.reshape(C * FH * FW, self.x_shape[0], -1).transpose(0, 1)
        dx = col2im(dcol, self.x_shape, FH, FW, self.stride, self.pad)

        return dx
