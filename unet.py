import torch
import torch.nn as nn


class CBR(nn.Module):
    """
    [정밀 분석 단계]
    구조: Conv(S=1) -> BN -> ReLU -> Conv(S=1) -> BN -> ReLU
    역할: 해상도를 유지하며 특징을 깊게 추출 (Skip Connection용 데이터 생성)
    """

    def __init__(self, in_channels, out_channels):
        super(CBR, self).__init__()
        self.layer = nn.Sequential(
            # 첫 번째 합성곱 (특징 추출 1차)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 두 번째 합성곱 (특징 추출 2차 & 비선형성 강화)
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class DownSample(nn.Module):
    """
    [압축 및 요약 단계]
    구조: Conv(S=2) -> BN -> ReLU
    역할: 해상도를 절반으로 줄이면서(Downsampling) 정보를 요약
    """

    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            # Stride=2 합성곱 (학습 가능한 다운샘플링)
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(out_channels),  # 데이터 분포 정규화 (Zero-point calibration)
            nn.ReLU(inplace=True),  # 노이즈 필터링
        )

    def forward(self, x):
        return self.layer(x)


class UNetDenoise(nn.Module):
    def __init__(self):
        super(UNetDenoise, self).__init__()

        # --- Encoder (Downsampling Path) ---
        # Level 1: Input(1) -> 64
        self.enc1 = CBR(1, 64)  # 분석: (1, 256, 256) -> (64, 256, 256) ★Skip1
        self.down1 = DownSample(64, 64)  # 압축: (64, 256, 256) -> (64, 128, 128)

        # Level 2: 64 -> 128
        self.enc2 = CBR(64, 128)  # 분석: (64, 128, 128) -> (128, 128, 128) ★Skip2
        self.down2 = DownSample(128, 128)  # 압축: (128, 128, 128) -> (128, 64, 64)

        # Level 3: 128 -> 256
        self.enc3 = CBR(128, 256)  # 분석: (128, 64, 64) -> (256, 64, 64) ★Skip3
        self.down3 = DownSample(256, 256)  # 압축: (256, 64, 64) -> (256, 32, 32)

        # Level 4: 256 -> 512
        self.enc4 = CBR(256, 512)  # 분석: (256, 32, 32) -> (512, 32, 32) ★Skip4
        self.down4 = DownSample(512, 512)  # 압축: (512, 32, 32) -> (512, 16, 16)

        # --- Bottleneck (Latent Space) ---
        # 가장 깊은 곳: 소리의 '의미(Context)'만 남은 상태
        self.bottleneck = CBR(512, 1024)  # (512, 16, 16) -> (1024, 16, 16)

        # --- Decoder (Upsampling Path) ---
        # Level 4 복원
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 2배 확대
        self.dec4 = CBR(1024, 512)  # Concat(512+512) -> 512

        # Level 3 복원
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = CBR(512, 256)  # Concat(256+256) -> 256

        # Level 2 복원
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = CBR(256, 128)  # Concat(128+128) -> 128

        # Level 1 복원 (최종 해상도)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = CBR(128, 64)  # Concat(64+64) -> 64

        # --- Final Output ---
        # 채널을 1개(dB값)로 줄임. (활성화 함수 없음: dB는 음수 가능)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

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

        return output


# --- 모델 검증용 코드 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDenoise().to(device)

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
