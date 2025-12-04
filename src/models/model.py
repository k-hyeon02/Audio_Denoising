# UNet, test 정의

import torch
from models.components.module import Module
import models.components.DoubleConv as DoubleConv


class UNet(Module):
    pass


def test():
    x = torch.randn((3, 1, 161, 161))
    model = UNet(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
