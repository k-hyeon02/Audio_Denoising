import torch
from layers import *

UNET_CHANNELS = [1, 32, 64, 128, 256, 512]

class UNet:
    def __init__(self, channels=None, filter_size=3):
        if channels is None:
            channels = UNET_CHANNELS

        c_in, c1, c2, c3, c4, c_bot = channels

        self.layers = {}

        # ==========================================
        # [Encoder] Downsampling Path (4 Layers)
        # 구조: DoubleConv : Conv -> act -> Conv -> act
        # -> Conv(stride=2, pad=0)
        # ==========================================

        # Layer 1 : input(1ch) -> 32ch (256*256 -> 128*128)
        self.enc1 = DoubleConv(c_in, c1, filter_size)
        # 2*2 Conv stride 2
        self.down1 = Conv2d(he_init(c1*filter_size*filter_size, (c1, c1, filter_size, filter_size)), 
                            torch.zeros(c1), stride=2, pad=1)

        # Layer 2 : 32 -> 64 (128*128 -> 64*64)
        self.enc2 = DoubleConv(c1, c2, filter_size)
        self.down2 = Conv2d(he_init(c2*filter_size*filter_size, (c2, c2, filter_size, filter_size)), 
                            torch.zeros(c2), stride=2, pad=1)

        # Layer 3 : 64 -> 128 (64*64 -> 32*32)
        self.enc3 = DoubleConv(c2, c3, filter_size)
        self.down3 = Conv2d(he_init(c3*filter_size*filter_size, (c3, c3, filter_size, filter_size)), 
                            torch.zeros(c3), stride=2, pad=1)

        # Layer 4 : 128 -> 256 (32*32 -> 16*16)
        self.enc4 = DoubleConv(c3, c4, filter_size)
        self.down4 = Conv2d(he_init(c4*filter_size*filter_size, (c4, c4, filter_size, filter_size)), 
                            torch.zeros(c4), stride=2, pad=1)

        # ==========================================
        # 2. Bottleneck
        # ==========================================
        self.bottleneck = DoubleConv(c4, c_bot, filter_size)

        # ==========================================
        # 3. Decoder (Upsampling Path)
        # 구조: Convtranspose2d -> Concat -> DoubleConv
        # ==========================================

        # Layer 4 : 512 -> 256
        self.up4 = ConvTransposed2d(he_init(c_bot*filter_size*filter_size, (c_bot, c4, filter_size, filter_size)), 
                                    torch.zeros(c4), pad=1, output_pad=1)
        self.cat4 = Concat()
        # 입력 채널 = up(128) + skip(128) = 256
        self.dec4 = DoubleConv(2*c4, c4, filter_size)

        # Layer 3 : 256 -> 128
        self.up3 = ConvTransposed2d(he_init(c4*filter_size*filter_size, (c4, c3, filter_size, filter_size)), 
                                    torch.zeros(c3), pad=1, output_pad=1)
        self.cat3 = Concat()
        self.dec3 = DoubleConv(2*c3, c3, filter_size)

        # Layer 2 : 128 -> 64
        self.up2 = ConvTransposed2d(he_init(c3*filter_size*filter_size, (c3, c2, filter_size, filter_size)), 
                                    torch.zeros(c2), pad=1, output_pad=1)
        self.cat2 = Concat()
        self.dec2 = DoubleConv(2*c2, c2, filter_size)

        # Layer 1 : 64 -> 32
        self.up1 = ConvTransposed2d(he_init(c2*filter_size*filter_size, (c2, c1, filter_size, filter_size)), 
                                    torch.zeros(c1), pad=1, output_pad=1)
        self.cat1 = Concat()
        self.dec1 = DoubleConv(2*c1, c1, filter_size)

        # ==========================================
        # 4. Final Output
        # 32ch -> Out_channels (1x1 Conv)
        # ==========================================
        self.final_conv = Conv2d(he_init(c1, (c_in, c1, 1, 1)), torch.zeros(1), stride=1, pad=0)

        # 학습 가능한 모든 모듈 리스트 (step 및 backward 관리용)
        self.modules = [
            self.enc1, self.down1, self.enc2, self.down2, 
            self.enc3, self.down3, self.enc4, self.down4,
            self.bottleneck,
            self.up4, self.dec4, self.up3, self.dec3,
            self.up2, self.dec2, self.up1, self.dec1,
            self.final_conv
        ]

        # Concat 레이어 (backward시 필요)
        self.cats = [self.cat4, self.cat3, self.cat2, self.cat1]
        self.ups = [self.up4, self.up3, self.up2, self.up1]

    def forward(self, x):
        # --- Encoder ---
        s1 = self.enc1.forward(x)  # Skip 1
        p1 = self.down1.forward(s1)

        s2 = self.enc2.forward(p1)  # Skip 2
        p2 = self.down2.forward(s2)

        s3 = self.enc3.forward(p2)  # Skip 3
        p3 = self.down3.forward(s3)

        s4 = self.enc4.forward(p3)  # Skip 4
        p4 = self.down4.forward(s4)

        # --- Bottleneck ---
        b = self.bottleneck.forward(p4)

        # --- Decoder ---
        # Layer 4
        d4 = self.up4.forward(b)
        d4 = self.cat4.forward(d4, s4)  # Concat with Skip 4
        d4 = self.dec4.forward(d4)

        # Layer 3
        d3 = self.up3.forward(d4)
        d3 = self.cat3.forward(d3, s3)  # Concat with Skip 3
        d3 = self.dec3.forward(d3)

        # Layer 2
        d2 = self.up2.forward(d3)  
        d2 = self.cat2.forward(d2, s2)  # Concat with Skip 2
        d2 = self.dec2.forward(d2)

        # Layer 1
        d1 = self.up1.forward(d2)
        d1 = self.cat1.forward(d1, s1)  # Concat with Skip 1
        d1 = self.dec1.forward(d1)

        out = self.final_conv.forward(d1)

        return out

    def backward(self, dout):
        dout = self.final_conv.backward(dout)

        # Decoder 1
        dout = self.dec1.backward(dout)
        dout_up, dout_skip = self.cat1.backward(dout)  # Split
        dout = self.up1.backward(dout_up)
        dout_s1 = dout_skip

        # Decoder 2
        dout = self.dec2.backward(dout)
        dout_up, dout_skip = self.cat2.backward(dout)
        dout = self.up2.backward(dout_up)
        dout_s2 = dout_skip

        # Decoder 3
        dout = self.dec3.backward(dout)
        dout_up, dout_skip = self.cat3.backward(dout)
        dout = self.up3.backward(dout_up)
        dout_s3 = dout_skip

        # Decoder 4
        dout = self.dec4.backward(dout)
        dout_up, dout_skip = self.cat4.backward(dout)
        dout = self.up4.backward(dout_up)
        dout_s4 = dout_skip

        # Bottleneck
        dout = self.bottleneck.backward(dout)

        # Encoder 4
        dout = self.down4.backward(dout)
        dout = dout + dout_s4  # Gradient Sum
        dout = self.enc4.backward(dout)

        # Encoder 3
        dout = self.down3.backward(dout)
        dout = dout + dout_s3
        dout = self.enc3.backward(dout)

        # Encoder 2
        dout = self.down2.backward(dout)
        dout = dout + dout_s2
        dout = self.enc2.backward(dout)

        # Encoder 1
        dout = self.down1.backward(dout)
        dout = dout + dout_s1
        dout = self.enc1.backward(dout)

        return dout

    def step(self, lr=0.01):
        # 모든 하위 모듈의 가중치 갱신
        for module in self.modules:
            module.step(lr)
