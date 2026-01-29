import torch
import numpy as np
from lietorch import SE3
from src.model import VO  # VO 클래스가 정의된 파일 경로
from utils.geo_utils import *


def test_full_vo_pipeline():
    print("--- [Verification] Full VO Pipeline Integration Test ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 모델 설정
    cfg = {} # 필요한 설정값들
    model = VO(cfg).to(device)
    
    # 2. 가상 데이터셋 준비 (배치=1)
    B, N = 1, 800
    batch = {
        'kpts': torch.randn(B, N, 2, device=device),
        'pts_3d': torch.randn(B, N, 3, device=device) + torch.tensor([0,0,5.0], device=device),
        'descs': torch.randn(B, N, 256, device=device),
        'kpts_tp1': torch.randn(B, N, 2, device=device),
        'tri_indices': [torch.randint(0, N, (300, 3), device=device)],
        'mask': torch.ones(B, N, device=device),
        'calib': torch.tensor([[718., 718., 607., 307.]], device=device)
    }

    # 3. Forward 실행
    print(f"Running VO forward with 8 iterations...")
    try:
        # predictions는 리스트이며, 각 원소는 {'pose': [B, 4, 4], 'depth': [B, N, 1]}
        predictions = model(batch, iters=8, mode='train')
        
        # 4. 결과 검증
        print(f"Total predictions stored: {len(predictions)}")
        
        # 마지막 이터레이션의 결과 확인
        final_pose = predictions[-1]['pose']
        final_depth = predictions[-1]['depth']
        
        print(f"Final Pose Shape: {final_pose.shape} (Expected: [1, 4, 4])")
        print(f"Final Depth Shape: {final_depth.shape} (Expected: [1, 800, 1])")
        
        # 포즈가 업데이트 되었는지 확인 (Identity가 아님)
        is_updated = not torch.allclose(final_pose, torch.eye(4, device=device).unsqueeze(0))
        print(f"Pose was updated from Identity: {is_updated}")

        print("\n✅ [Success] VO 파이프라인 전체 루프가 성공적으로 실행되었습니다.")
        
    except Exception as e:
        print(f"\n❌ [Fail] 파이프라인 실행 중 에러 발생:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_vo_pipeline()