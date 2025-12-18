from layers import *

import torch
import torch.nn as nn
import torch.optim as optim


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, pad=1, device=None):
        super().__init__()
        self.DC = nn.Sequential(
            Conv2d(in_channels, out_channels, stride=stride, pad=pad, device=device),
            # BatchNorm(in_channels),
            nn.ReLU(),
            Conv2d(out_channels, out_channels, stride=stride, pad=pad, device=device),
            # BatchNorm(in_channels),
            nn.ReLU()
        )
        self.skip = None

    def forward(self, x):
        x = self.DC.forward(x)
        self.x = x
        return x
    
    # def backward(self, dout):
    #     dout = self.DC.backward(dout)
    #     return dout


class Down(nn.Module):     # 내려오고 doubleconv 진행
    def __init__(self, in_channels, out_channels, stride=1, pad=1, device=None):
        super().__init__()
        self.SConv = Conv2d(in_channels, out_channels, stride=2, pad=1, device=device)
        self.DC = DoubleConv(out_channels, out_channels, stride=stride, pad=pad, device=device)

    def forward(self, x):
        x = self.SConv.forward(x)
        x = self.DC.forward(x)

        return x
    
    # def backward(self, dout):       # 이 dout은 up에서 떨어져 나온 오차까지 받은 값
    #     dout = self.DC.backward(dout)
    #     dout = self.SConv.backward(dout)
    #     return dout

class Up(nn.Module):       # 올라가고 doubleconv 진행
    def __init__(self, in_channels, out_channels, stride=1, pad=1, device=None):
        super().__init__()
        self.TConv = ConvTranspose2d(in_channels, out_channels, stride=2, pad=1, device=device)
        self.DC = DoubleConv(in_channels, out_channels, stride=1, pad=1, device=device)
        self.sc = None

    def forward(self, x, sc):
        x = self.TConv.forward(x)
        out = torch.cat((sc, x), axis=1)  # 채널 방향으로 연결 
        out = self.DC.forward(out)

        return out
    
    # def backward(self, dout):
    #     dout = self.DC.backward(dout)
    #     _, C, __, ___ = dout.shape
    #     dsc = dout[:, :C//2, :, :]
    #     dout = self.TConv.backward(dout[:, C//2:, :, :])

    #     return dsc, dout        # 하나는 인코더로, 하나는 아래로 흘려보낸다.


CHANNELS = [1, 64, 128, 256, 512, 1024]


class UNet(nn.Module):
    def __init__(self, channels, device=None):
        super().__init__()
        self.U_down = nn.ModuleList([
            DoubleConv(channels[0], channels[1], device=device),
            Down(channels[1], channels[2], device=device),
            Down(channels[2], channels[3], device=device),
            Down(channels[3], channels[4], device=device),
        ])

        self.bottleneck = Down(channels[4], channels[5], device=device)

        self.U_up = nn.ModuleList([
            Up(channels[5], channels[4], device=device),
            Up(channels[4], channels[3], device=device),
            Up(channels[3], channels[2], device=device),
            Up(channels[2], channels[1], device=device)
        ])
        self.outlayer = Conv2d(channels[1], channels[0], kernel_size=1, pad=0, device=device)

    def forward(self, x):
        self.skip = []
        i = 3
        for module in self.U_down:
            x = module.forward(x)
            self.skip.append(x)

        x = self.bottleneck.forward(x)

        for module in self.U_up:
            x = module.forward(x, self.skip[i])
            i -= 1
        out = self.outlayer.forward(x)

        return out

    # def backward(self, dout):
    #     self.skip = []
    #     i = 3
    #     dout = self.outlayer.backward(dout)
    #     for module in reversed(self.U_up.modules):
    #         dsc, dout = module.backward(dout)
    #         self.skip.append(dsc)

    #     dout = self.bottleneck.backward(dout)

    #     for module in reversed(self.U_down.modules):
    #         dout = module.backward(dout + self.skip[i])
    #         i -= 1
    #     return dout

x = torch.randn(3, 1, 128, 128)
def test():


    loss_fn = F.l1_loss

    model = UNet(CHANNELS)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    preds = model(x)
    print(preds.shape)
    # loss_func = torch.nn.MSELoss()
    # loss = loss_func.forward(preds, x)
    loss = loss_fn(preds, x)
    print(loss.item())
    before = model.U_down[0].DC[0].W.clone()
    loss.backward()
    optimizer.step()
    after = model.U_down[0].DC[0].W

    print(torch.equal(before, after))  # False여야 정상



if __name__ == "__main__":
    for i in range(10):
        test()