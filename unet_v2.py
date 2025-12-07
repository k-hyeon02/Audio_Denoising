from layers import *
from components.tools import *

class DoubleConv(Module):
    def __init__(self, in_channels, out_channels, stride=1, pad=1):
        super().__init__()
        self.DC = Sequential(
            Conv2d(in_channels, out_channels, stride=stride, pad=pad),
            # BatchNorm(in_channels),
            ReLU(),
            Conv2d(out_channels, out_channels, stride=stride, pad=pad),
            # BatchNorm(in_channels),
            ReLU()
        )
        self.skip = None

    def forward(self, x):
        x = self.DC.forward(x)
        self.x = x
        return x
    
    def backward(self, dout):
        dout = self.DC.backward(dout)
        return dout
    

class Down(Module):     # 내려오고 doubleconv 진행
    def __init__(self, in_channels, out_channels, stride=1, pad=1):
        super().__init__()
        self.SConv = Conv2d(in_channels, out_channels, stride=2, pad=1)
        self.DC = DoubleConv(out_channels, out_channels, stride=stride, pad=pad)

    def forward(self, x):
        x = self.SConv.forward(x)
        x = self.DC.forward(x)

        return x
    
    def backward(self, dout):       # 이 dout은 up에서 떨어져 나온 오차까지 받은 값
        dout = self.DC.backward(dout)
        dout = self.SConv.backward(dout)
        return dout

class Up(Module):       # 올라가고 doubleconv 진행
    def __init__(self, in_channels, out_channels, stride=1, pad=1):
        super().__init__()
        self.TConv = ConvTranspose2d(in_channels, out_channels, stride=2, pad=1)
        self.DC = DoubleConv(in_channels, out_channels, stride=1, pad=1)
        self.sc = None

    def forward(self, x, sc):

        x = self.TConv.forward(x)
        out = torch.cat((sc, x), axis=1)  # 채널 방향으로 연결 
        out = self.DC.forward(out)

        return out
    
    def backward(self, dout):
        dout = self.DC.backward(dout)
        _, C, __, ___ = dout.shape
        dsc = dout[:, :C//2, :, :]
        dout = self.TConv.backward(dout[:, C//2:, :, :])

        return dsc, dout        # 하나는 인코더로, 하나는 아래로 흘려보낸다.


CHANNELS = [1, 64, 128, 256, 512, 1024]


class UNet(Module):
    def __init__(self, channels):
        self.U_down = ModuleList([
            DoubleConv(1, 64),
            Down(64, 128),
            Down(128, 256),
            Down(256, 512)
        ])
        self.bottleneck = Down(512, 1024)
        self.U_up = ModuleList([
            Up(1024, 512),
            Up(512, 256),
            Up(256, 128),
            Up(128, 64)
        ])
        self.outlayer = Conv2d(64, 1, kernel_size=1, pad=0)

    def forward(self, x):
        self.skip = []
        i = 3
        for module in self.U_down.modules:
            x = module.forward(x)
            self.skip.append(x)
        
        x = self.bottleneck.forward(x)

        for module in self.U_up.modules:
            x = module.forward(x, self.skip[i])
            i -= 1
        out = self.outlayer.forward(x)

        return out
    
    def backward(self, dout):
        self.skip = []
        i = 3
        dout = self.outlayer.backward(dout)
        for module in reversed(self.U_up.modules):
            dsc, dout = module.backward(dout)
            self.skip.append(dsc)
        
        dout = self.bottleneck.backward(dout)

        for module in reversed(self.U_down.modules):
            dout = module.backward(dout + self.skip[i])
            i -= 1
        return dout




def test():
    x = torch.randn(3, 1, 128, 128)
    model = UNet(CHANNELS)
    preds = model.forward(x)
    print(model.parameters)
    print(preds.shape)
    # loss_func = torch.nn.MSELoss()
    # loss = loss_func.forward(preds, x)
    out = model.backward(x)
    print(out.shape)

if __name__ == "__main__":
    test()
