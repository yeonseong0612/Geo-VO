import torch
import numpy as np
from src.layer import CyclicErrorModule
from src.loader import DataFactory
from lietorch import SE3

def test_cyclic_unit():
    # 1. 테스트용 최소 Config 설정
    class DummyConfig:
        proj_home = "/home/yskim/projects/Geo-VO/"
        odometry_home = "/home/yskim/projects/vo-labs/data/kitti_odometry/" # 실제 데이터 경로로 수정 필요
        traintxt = "train.txt"
        trainsequencelist = ["00"]
        color_subdir = "datasets/sequences/"
        poses_subdir = "poses/"
        calib_subdir = "datasets/sequences/"
    
    cfg = DummyConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    # 2. 모듈 초기화
    cyclic_module = CyclicErrorModule(baseline=0.54).to(device)

    # 3. 데이터 로더에서 실제 샘플 하나 가져오기
    try:
        dataset = DataFactory(cfg, mode='train')
        sample = dataset[0]
        
        # 실제 데이터에서 값 추출
        rel_pose = sample['rel_pose'].to(device) # SE3 객체
        intrinsics = sample['intrinsics'].unsqueeze(0).to(device) # [1, 4]
        
        # 입력 텐서 생성 (B=1, N=1000)
        kpts_Lt = torch.randn(1, 1000, 2).to(device)
        depth_Lt = torch.ones(1, 1000, 1).to(device) * 5.0

        print(f"\n--- Input Dimensions ---")
        print(f"kpts_Lt   : {kpts_Lt.shape}")
        print(f"rel_pose  : {rel_pose.data.shape} (SE3)")
        print(f"intrinsics: {intrinsics.shape}")

        # 4. 모듈 실행
        e_proj = cyclic_module(kpts_Lt, depth_Lt, rel_pose, intrinsics)
        
        print(f"\n--- Output Results ---")
        print(f"e_proj shape: {e_proj.shape}")
        
        if e_proj.dim() == 3:
            print("✅ Result: [B, N, 2] - Correct for UpdateBlock!")
        elif e_proj.dim() == 4:
            print("⚠️ Result: [B, 1, N, 2] - Needs Squeeze(1) before UpdateBlock!")

    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        # 구체적인 에러 위치 확인을 위해 traceback 출력
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_cyclic_unit()