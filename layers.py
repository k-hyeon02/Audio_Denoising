import torch
from utils import im2col

# ex) (1, 1, 256, 256)
class Conv2d:
    def __init__(self, W, b, stride=1, pad=1):
        self.W = W  # (16, 1, 3, 3)
        self.b = b  # (32,)
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        # 출력 크기 계산
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2 * self.pad - FW) // self.stride + 1

        # im2col 실행
        col = im2col(x, FH, FW, self.stirde, self.pad)  # (1, 9, 256*256)

        # 필터 전개
        col_W = self.W.reshape(FN, -1)  # (32, 9)

        # 행렬 곱 + bias
        # (32, 9) * (1, 9, 256*256) = (1, 32, 256*256)
        # (1, 32, 256*256) + (32, 1)
        out = torch.matmul(col_W, col) + self.b.reshape(-1,1)

        return out
