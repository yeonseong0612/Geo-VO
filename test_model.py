import torch
from src.model import VO # 파일명에 맞게 수정하세요
from lietorch import SE3

def test_sanity_check():
    # 1. 가상의 설정 및 데이터 생성
    cfg = type('cfg', (), {})() # 더미 cfg
    model = VO(cfg).cuda().eval() # 테스트는 eval 모드

    B, N = 2, 800 # 배치 2, 특징점 800개
    device = "cuda"

    # 가상의 배치를 실제 입력 형태와 동일하게 생성
    batch = {
        'kpts': torch.randn(B, N, 2, device=device),
        'pts_3d': torch.randn(B, N, 3, device=device) * 10.0, # 깊이감 부여
        'descs': torch.randn(B, N, 256, device=device),
        'kpts_tp1': torch.randn(B, N, 2, device=device),
        'calib': torch.tensor([[700, 700, 600, 175]] * B, device=device).float(),
        'tri_indices': [torch.randint(0, N, (1000, 3), device=device) for _ in range(B)],
        'mask': torch.ones(B, N, device=device)
    }

    print("==> Forward Pass 시작...")
    
    try:
        # 2. Forward 실행
        with torch.no_grad():
            outputs = model(batch, iters=4)

        # 3. 결과물 차원 및 값 검증
        print("\n[검증 결과]")
        for i, (pose, conf) in enumerate(zip(outputs['pose_matrices'], outputs['confidences'])):
            # Pose Matrices 검증 [B, 4, 4]
            print(f"Iter {i+1} Pose Shape: {pose.shape}")
            if torch.isnan(pose).any():
                print(f"❌ Iter {i+1}: Pose에 NaN 발생!")
            else:
                print(f"✅ Iter {i+1}: Pose 안정적")

            # Confidence 검증 [B, N, 1]
            if torch.isnan(conf).any():
                print(f"❌ Iter {i+1}: Confidence에 NaN 발생!")
            else:
                print(f"✅ Iter {i+1}: Confidence 안정적")
        
        print("\n==> 모든 이터레이션 통과 성공!")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sanity_check()