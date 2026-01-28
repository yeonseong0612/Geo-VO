import numpy as np
import os

def verify_geo_vo_precomputed(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 1. 데이터 로드 (mmap_mode='r'로 효율적인 읽기)
    data = np.load(file_path, allow_pickle=True)
    
    print(f"\n{'='*20} 📂 파일 정보: {os.path.basename(file_path)} {'='*20}")
    
    # 2. 저장된 모든 키(Keys) 확인
    for key in data.files:
        val = data[key]
        print(f"✅ {key:<12} : Shape {str(val.shape):<15} | dtype: {val.dtype}")

    # 3. 데이터 무결성 및 기하 정보 상세 검사
    print(f"\n{'*'*20} 🔍 데이터 무결성 체크 {'*'*20}")
    
    # [A] 마스크 및 특징점 유효성
    mask = data['mask']
    kpts = data['kpts']
    num_valid = np.sum(mask)
    print(f"⭐ 유효 특징점 수    : {num_valid} / {len(mask)} (Masked)")

    # [B] 카메라 내적 행렬 (K) 및 주점 보정 확인
    K = data['K']
    img_sz = data['image_size'] # [H, W] -> [352, 1216]
    cx, cy = K[0, 2], K[1, 2]
    print(f"⭐ 보정된 주점(cx, cy): ({cx:.2f}, {cy:.2f})")
    print(f"⭐ 이미지 규격(H, W)  : {img_sz[0]} x {img_sz[1]}")
    
    # cy가 리사이즈/크롭 후 이미지 중심 근처에 있는지 체크 (보통 352/2 = 176 근처)
    if 150 < cy < 200:
        print(f"⭐ 주점 보정 상태     : PASS (cy가 {cy:.1f}로 정상 범위 내에 있음)")
    else:
        print(f"⭐ 주점 보정 상태     : WARNING (cy 위치 확인 필요)")

    # [C] 삼각형(DT) 정보 검사
    tri_idx = data['tri_indices']
    if tri_idx.size > 0:
        print(f"⭐ 생성된 삼각형 수   : {len(tri_idx)}개")
        # 인덱스 유효성: 모든 삼각형 정점이 유효 특징점 범위 내에 있는지
        is_tri_valid = np.max(tri_idx) < num_valid
        print(f"⭐ 삼각형 인덱스 유효 : {'PASS' if is_tri_valid else 'FAIL'}")
    else:
        print("⚠️ 생성된 삼각형이 없습니다. (특징점 부족 가능성)")

    # [D] 디스크립터 정밀도 확인
    descs = data['descs']
    if descs.dtype == np.float16:
        print(f"⭐ 데이터 압축 상태   : PASS (fp16 적용됨)")
    else:
        print(f"⭐ 데이터 압축 상태   : NOTE (fp32 사용 중)")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    # 전처리 결과가 저장된 실제 경로로 수정하세요.
    SAMPLE_PATH = "/home/jnu-ie/kys/Geo-VO/gendata/precomputed/00/image_2/000120.npz"
    
    verify_geo_vo_precomputed(SAMPLE_PATH)