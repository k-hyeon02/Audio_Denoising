import torch
import torch.nn as nn


class CBR(nn.Module):
    """
    구조: Conv(S=1) -> LeakyReLU -> Conv(S=1) -> LeakyReLU
    """

    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.layer = nn.Sequential(
            # 첫 번째 합성곱 (특징 추출 1차)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
            # 두 번째 합성곱 (특징 추출 2차 & 비선형성 강화)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    """
    구조: Conv(S=2) -> LeakyReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # Stride=2 합성곱
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(inplace=True),  # 노이즈 필터링
        )

    def forward(self, x):
        return self.layer(x)


UNET_CHANNELS = [1, 32, 64, 128, 256, 512]

class UNet(nn.Module):

    def __init__(self, channels=None):
        super(UNet, self).__init__()

        channels = UNET_CHANNELS

        c_in, c1, c2, c3, c4, c_bot = channels

        # --- Encoder (Downsampling Path) ---
        # Level 1: Input(1) -> 32
        self.enc1 = CBR(c_in, c1)  # 분석: (1, 128, 128) -> (32, 128, 128) ★Skip1
        self.down1 = DownSample(c1, c1)  # 압축: (32, 128, 128) -> (32, 64, 64)

        # Level 2: 32 -> 64
        self.enc2 = CBR(c1, c2)  # 분석: (32, 64, 64) -> (64, 64, 64) ★Skip2
        self.down2 = DownSample(c2, c2)  # 압축: (64, 64, 64) -> (64, 32, 32)

        # Level 3: 64 -> 128
        self.enc3 = CBR(c2, c3)  # 분석: (64, 32, 32) -> (128, 32, 32) ★Skip3
        self.down3 = DownSample(c3, c3)  # 압축: (128, 32, 32) -> (128, 32, 32)

        # Level 4: 128 -> 256
        self.enc4 = CBR(c3, c4)  # 분석: (128, 32, 32) -> (256, 32, 32) ★Skip4
        self.down4 = DownSample(c4, c4)  # 압축: (256, 32, 32) -> (256, 16, 16)

        # --- Bottleneck (Latent Space) ---
        # 가장 깊은 곳: 소리의 '의미(Context)'만 남은 상태
        self.bottleneck = CBR(c4, c_bot)  # (256, 16, 16) -> (512, 16, 16)

        # --- Decoder (Upsampling Path) ---
        # Level 4 복원
        self.up4 = nn.ConvTranspose2d(c_bot, c4, kernel_size=2, stride=2)  # 2배 확대
        self.dec4 = CBR(c_bot, c4)  # Concat(256+256) -> 256

        # Level 3 복원
        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = CBR(c4, c3)  # Concat(128+128) -> 128

        # Level 2 복원
        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = CBR(c3, c2)  # Concat(64+64) -> 64

        # Level 1 복원 (최종 해상도)
        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = CBR(c2, c1)  # Concat(32+32) -> 32

        # --- Final Output ---
        # 채널을 1개(dB값)로 줄임.
        self.final = nn.Conv2d(c1, c_in, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # === Encoder ===
        # Level 1
        s1 = self.enc1(x)  # Skip Connection 저장
        p1 = self.down1(s1)  # Downsampling

        # Level 2
        s2 = self.enc2(p1)
        p2 = self.down2(s2)

        # Level 3
        s3 = self.enc3(p2)
        p3 = self.down3(s3)

        # Level 4
        s4 = self.enc4(p3)
        p4 = self.down4(s4)

        # === Bottleneck ===
        b = self.bottleneck(p4)

        # === Decoder ===
        # Level 4
        u4 = self.up4(b)
        # Skip Connection 결합: (UpFeature + SkipFeature)
        cat4 = torch.cat((u4, s4), dim=1)
        d4 = self.dec4(cat4)

        # Level 3
        u3 = self.up3(d4)
        cat3 = torch.cat((u3, s3), dim=1)
        d3 = self.dec3(cat3)

        # Level 2
        u2 = self.up2(d3)
        cat2 = torch.cat((u2, s2), dim=1)
        d2 = self.dec2(cat2)

        # Level 1
        u1 = self.up1(d2)
        cat1 = torch.cat((u1, s1), dim=1)
        d1 = self.dec1(cat1)

        # Output
        output = self.final(d1)
        output = self.sigmoid(output)

        return output


# --- 모델 검증용 코드 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    # 더미 데이터 생성 (배치크기: 4, 채널: 1, 높이: 256, 너비: 256)
    dummy_input = torch.randn(4, 1, 256, 256).to(device)

    print(
        f"모델 파라미터 개수: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print(f"입력 데이터 형태: {dummy_input.shape}")

    output = model(dummy_input)
    print(f"출력 데이터 형태: {output.shape}")

    # 차원 검증
    if output.shape == dummy_input.shape:
        print("✅ 검증 성공: 입력과 출력의 차원이 완벽하게 일치합니다.")
    else:
        print("❌ 검증 실패: 차원이 일치하지 않습니다.")
