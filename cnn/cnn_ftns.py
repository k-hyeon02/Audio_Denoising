import numpy as np

# im2col
def im2col(input_data, filter_h, filter_w, stride=1,  pad=0):
    """
    입력 데이터를 필가 적용될 영역마다 잘라서 열(column)로 전개하는 함수
    
    Parameters:
        Input_data (batch_size, n_channel, h, w) : 4_dimension
        filter_h, filter_w : 필터 높이, 너비
    
    Returns:
        (윈도우 총 개수, 윈도우 하나의 크기)

    """
    
    N, C, H, W = input_data.shape

    # 1. 출력 크기 계산
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # 2. 패딩 적용
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # 3. 거대 행렬을 위한 공간 생성
    # 차원 : (총 윈도우 개수, 필터영역크기) 이 모양은 함수의 마지막에 만들 예정
    # 필터영역크기 = 채널 * 필터높이 * 필터너비
    # 총 윈도우 개수 = 배치 * 출력높이 * 출력너비
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    # 4. 데이터 채워넣기 (슬라이딩 윈도우 영역을 복사)
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w

            # 이미지에서 x, y 부터 x_max, y_max까지 stride 간격으로 필셀을 가져온다.
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    
    # 5. 모양 바꾸기 (reshape & transpose)
    # (N, C, filter_h, filter_w, out_h, out_w)
    # -> (N, out_h, out_w, C, filter_h, filter_w) 로 축 변경
    col = col.transpose(0, 4, 5, 1, 2, 3)

    # 최종적으로 2차원 행렬로 펼친다.
    # 행: 윈도우 하나 (이미지 조각)/ 열: 픽셀 값들
    col = col.reshape(N * out_h * out_w, -1)
    
    return col

# cnn
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # 필터 (Weights) : (필터개수 FN, 채널 C, FH, FW)
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape

        # 1. 출력 크기 계산
        out_h = (H + 2*self.pad - FH) // self.stride + 1
        out_w = (W + 2*self.pad - FW) // self.stride + 1

        # 2. image to column
        col = im2col(x, FH, FW, self.stride, self.pad)

        # 3. filter to 2D
        # (FN, C, FH, FW) -> (FN, C*FH*FW)
        col_W = self.W.reshape(FN, -1).T    # shape : (필터 크기, 필터 개수)

        # 4. (데이터 개수, 필터 크기) @ (필터 크기, 필터 개수) = (데이터 개수, 필터 개수)
        out = col @ col_W + self.b

        # 5. Reshape to 4D img
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
        