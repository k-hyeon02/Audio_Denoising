import os
import torchaudio
import soundata

# ---------------------------------------------------------
# [핵심] 경로 안전하게 설정하기 (절대 경로 변환)
# ---------------------------------------------------------

# 1. 현재 이 파이썬 파일이 있는 폴더의 절대 경로를 구합니다.
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 원하는 데이터 저장 위치 설정
# 예: 현재 폴더와 같은 위치에 'data' 폴더 생성 (상위 폴더로 가고 싶으면 ".." 추가)
# data_root = C:/Users/.../Project/data 가 됩니다.
data_root = os.path.join(current_dir, "data") 

# 3. 폴더가 없으면 미리 만듭니다. (이게 없으면 종종 에러 남)
if not os.path.exists(data_root):
    os.makedirs(data_root)
    print(f"폴더를 생성했습니다: {data_root}")
else:
    print(f"기존 폴더를 사용합니다: {data_root}")

# ---------------------------------------------------------
# [다운로드] 수정된 경로 변수(data_root) 사용
# ---------------------------------------------------------

print("\n1. Librispeech 다운로드 중...")
try:
    dataset_librispeech = torchaudio.datasets.LIBRISPEECH(
        root=data_root,  # 수정된 경로 사용
        url="train-clean-100", 
        download=True
    )
    print("✅ Librispeech 완료")
except Exception as e:
    print(f"❌ Librispeech 오류: {e}")

print("\n2. UrbanSound8K 다운로드 중...")
try:
    # UrbanSound는 별도 폴더를 하나 더 만들어주는 게 관리가 편합니다.
    urban_path = os.path.join(data_root, "noise_datasets")
    
    dataset_urbansound = soundata.initialize("urbansound8k", data_home=urban_path)
    dataset_urbansound.download()
    dataset_urbansound.validate()
    print("✅ UrbanSound8K 완료")
except Exception as e:
    print(f"❌ UrbanSound8K 오류: {e}")