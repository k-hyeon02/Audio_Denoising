import numpy as np
import imcol as ic

    
class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, pad = 1):
        super().__init__()
        # 가중치 초기화 using He Initialization
        scale = np.sqrt(2.0 / (in_channels * kernel_size ** 2))
        self.params['W'] = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale # (FN, C, FH, FW)

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
        OW = (W + 2*self.pad - FW) // self.stride + 1   

        out = np.zeros((N, FN, OH, OW))

        # 2. image to column
        col = ic.im2col(x, FH, FW, self.stride, self.pad)  # (N*OH*OW, C*FH*FW)

        # 3. filter to 2D
        # (FN, C, FH, FW) -> (FN, C*FH*FW) -> (C*FH*FW, FN)
        col_W = self.W.reshape(FN, -1).T    # shape : (필터 크기, 필터 개수)

        # 4. (데이터 개수, 필터 크기) @ (필터 크기, 필터 개수) = (데이터 개수, 필터 개수) = (N*OH*OW, FN)
        out = col @ col_W + self.b      # ()@ (C*FH*FW, FN) -> 각 채널별로 계산한 값을 모두 더한다????

        # 5. Reshape to 4D img 
        out = out.reshape(N, OH, OW, -1).transpose(0, 3, 1, 2)    # (N, FN, OH, OW) 채널이 FN으로 바뀐다.

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
        dout =  dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # dW = x.T @ dout
        dW = np.dot(col.T, dout)
        # (C*FH*FW, FN) -> (FN, C, FH, FW)
        self.grads['W'] = dW.transpose(1, 0).reshape(FN, C, FH, FW)

        # dcol = dout @ W.T
        dcol = np.dot(dout, col_W.T)

        # col2im으로 이미지 크기 복원
        dx = ic.col2im(dcol, x.shape, FH, FW, self.stride, self.pad)
        
        return dx


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)    # 0 이하인 인덱스 저장
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        return dout