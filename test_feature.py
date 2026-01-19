import torch
import torch.nn as nn
from src.model import VO
from src.loader import DataFactory

def test_feature_dimensions():
    # 1. 테스트용 환경 설정 (사용자님의 경로에 맞게 설정됨)
    class DummyConfig:
        proj_home = "/home/yskim/projects/Geo-VO/"
        odometry_home = "/home/yskim/projects/vo-labs/data/kitti_odometry/"
        traintxt = "train.txt"
        trainsequencelist = ["00"]
        color_subdir = "datasets/sequences/"
        poses_subdir = "poses/"
        calib_subdir = "datasets/sequences/"
    cfg = DummyConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    # 2. 모델 및 데이터 로더 초기화
    model = VO(baseline=0.54).to(device)
    model.eval() # 추론 모드
    
    dataset = DataFactory(cfg, mode='train')
    sample = dataset[0]
    
    # 데이터 준비 및 GPU 이동
    images = sample['images'].unsqueeze(0).to(device)      # [1, 4, 3, H, W]
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device) # [1, 4]

    print(f"\n--- [Step 1] Feature Extraction Phase ---")
    with torch.no_grad():
        # Extractor 호출 (kpts, desc 2개를 리턴하도록 수정된 구조 기준)
        extraction_results = model.extractor(images[:, 0])
        print(f"Extractor returned {len(extraction_results)} values.")
        
        # 언패킹 (2개 혹은 3개 대응)
        if len(extraction_results) == 2:
            kpts_Lt, f_Lt = extraction_results
        else:
            kpts_Lt, f_Lt, _ = extraction_results

        # [핵심] 모든 데이터를 명시적으로 동일한 디바이스로 이동
        if isinstance(kpts_Lt, list):
            kpts_Lt = torch.stack([k.to(device) for k in kpts_Lt])
        else:
            kpts_Lt = kpts_Lt.to(device)

        if isinstance(f_Lt, list):
            f_Lt = torch.stack([f.to(device) for f in f_Lt])
        else:
            f_Lt = f_Lt.to(device)

        # SuperPoint의 경우 [B, 256, N]으로 나올 수 있으므로 [B, N, 256]으로 통일
        if f_Lt.dim() == 3 and f_Lt.shape[1] == 256:
            f_Lt = f_Lt.transpose(1, 2)

        print(f"kpts_Lt shape: {kpts_Lt.shape} (Device: {kpts_Lt.device})")
        print(f"f_Lt    shape: {f_Lt.shape} (Device: {f_Lt.device})")

    print(f"\n--- [Step 2] UpdateBlock Input Preparation ---")
    # UpdateBlock에 들어갈 4가지 재료 준비
    c_temp = f_Lt                                # [B, N, 256]
    c_stereo = f_Lt.clone()                      # [B, N, 256]
    e_proj = torch.randn_like(kpts_Lt).to(device) # [B, N, 2] 확실하게 GPU 이동
    c_geo = kpts_Lt                              # [B, N, 2]

    print(f"1. c_temp   : {c_temp.shape} on {c_temp.device}")
    print(f"2. c_stereo : {c_stereo.shape} on {c_stereo.device}")
    print(f"3. e_proj   : {e_proj.shape} on {e_proj.device}")
    print(f"4. c_geo    : {c_geo.shape} on {c_geo.device}")

    print(f"\n--- [Step 3] Final Concatenation Test ---")
    try:
        # 모든 텐서를 마지막 차원(dim=-1) 기준으로 합침
        combined = torch.cat([c_temp, c_stereo, e_proj, c_geo], dim=-1)
        
        print(f"✅ SUCCESS: Combined shape is {combined.shape}")
        print(f"Expected GRU Input Dimension: {combined.shape[-1]}")
        
        if combined.shape[-1] == 516: # (256 + 256 + 2 + 2)
            print(">>> Dimension match confirmed (516).")
        
    except RuntimeError as e:
        print(f"❌ FAILED: torch.cat failed!")
        print(f"Error Message: {e}")

if __name__ == "__main__":
    test_feature_dimensions()