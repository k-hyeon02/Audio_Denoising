import torch
import numpy as np
import os
from components.tools import Module


# 모델 gpu 이동 및 저장
def move_layer_to_device(layer, device):
    """레이어 내부의 W, b를 찾아 디바이스로 이동 (재귀적 순회)"""

    # 1. 기본 레이어 (Conv2d, ConvTransposed2d 등)
    if hasattr(layer, "params"):
        for key, tensor in layer.params.items():
            if tensor is not None and isinstance(tensor, torch.Tensor):
                if tensor.device != device:  # <--- 디버깅 조건 추가
                    print(
                        f"DEBUG: Moving {type(layer).__name__}.params['{key}'] from {tensor.device} to {device}"
                    )
                    layer.params[key] = tensor.to(device)

    # 2. 하위 모듈 (Sequential, DoubleConv, Down/Up 등)을 재귀적으로 순회
    # Module 클래스의 __dict__를 사용하여 모든 하위 모듈을 찾습니다.
    for value in layer.__dict__.values():
        if isinstance(value, Module):
            move_layer_to_device(value, device)


def move_model_to_device(model, device):
    """UNet 전체 파라미터 이동"""
    print(f"Moving model to {device}...")
    # UNet 자체가 커스텀 Module이므로, UNet 자체를 재귀 함수에 넣습니다.
    move_layer_to_device(model, device)


# 가중치 저장
def save_checkpoint(model, save_dir, filename):
    """UNet 모델의 모든 가중치를 찾아 저장 (재귀 순회)"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    checkpoint = {}

    # 재귀적으로 파라미터를 수집하는 헬퍼 함수
    def collect_params(module, prefix=""):

        # 1. 현재 모듈의 가중치 저장 (W, b)
        if hasattr(module, "params") and "W" in module.params:
            W = module.params["W"]
            # b가 있다면 b도 저장 (현재 Conv2d에는 b가 없으나 일반화를 위해 추가)
            b = module.params.get("b", None)

            checkpoint[prefix] = {
                "W": W.cpu(),
                "b": b.cpu() if b is not None else None,
            }

        # 2. 하위 모듈 재귀 순회 (Module 클래스의 __dict__ 이용)
        for name, value in module.__dict__.items():
            if isinstance(value, Module):
                # 하위 모듈 이름을 경로에 추가하여 재귀 호출
                collect_params(value, prefix=f"{prefix}/{name}")

    collect_params(model, prefix="UNet")
    torch.save(checkpoint, save_path)


# PSNR 계산 : PSNR = 10 * log10(MAX^2 / MSE)
def calculate_psnr(output, target):
    mse = torch.mean((output - target) ** 2)
    if mse == 0: # 둘이 완전히 똑같으면 무한대
        return 100.0
    
    # 데이터가 0~1 혹은 -1~1로 정규화되어 있다고 가정할 때 MAX=1.0으로 잡는 것이 일반적입니다.
    max_val = 1.0 
    psnr = 10 * torch.log10((max_val**2) / mse)
    return psnr
