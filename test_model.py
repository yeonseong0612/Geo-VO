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
        # test_model.py 수정 부분
        outputs = model(batch, iters=8, mode='train')

        # 이제 outputs는 딕셔너리입니다.
        predictions_pose = outputs['pose_matrices']
        predictions_conf = outputs['confidences']

        print(f"Total iterations stored: {len(predictions_pose)}")

        # 마지막 이터레이션의 포즈 가져오기
        final_pose = predictions_pose[-1] 
        print(f"Final Pose Shape: {final_pose.shape}")

        print("\n✅ [Success] VO 파이프라인 전체 루프가 성공적으로 실행되었습니다.")
        
    except Exception as e:
        print(f"\n❌ [Fail] 파이프라인 실행 중 에러 발생:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_full_vo_pipeline()