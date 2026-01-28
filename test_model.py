import torch
import torch.nn as nn
from src.model import VO
from lietorch import SE3

@torch.no_grad()
def create_dummy_batch(batch_size=2, num_kpts=800, num_tris=1200):
    """테스트를 위한 가상 데이터 생성"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch = {
        'kpts': torch.randn(batch_size, num_kpts, 2).to(device),
        'pts_3d': torch.rand(batch_size, num_kpts, 3).to(device) * 20.0 + 2.0, # Depth 2~22m
        'descs': torch.randn(batch_size, num_kpts, 256).to(device),
        'kpts_tp1': torch.randn(batch_size, num_kpts, 2).to(device),
        'calib': torch.tensor([[718.8, 718.8, 607.1, 185.2]] * batch_size).to(device),
        'mask': torch.ones(batch_size, num_kpts).bool().to(device),
        # 가변적인 삼각형 인덱스는 리스트로 처리
        'tri_indices': [torch.randint(0, num_kpts, (num_tris, 3)).to(device) for _ in range(batch_size)],
        'rel_pose': torch.randn(batch_size, 7).to(device) # GT Pose (Target)
    }
    return batch

def test_vo_forward_backward():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 테스트 시작 (Device: {device})")

    # 1. 모델 초기화
    # cfg 객체는 간단한 Namespace 등으로 대체 가능합니다.
    class DummyCfg:
        baseline = 0.54
    cfg = DummyCfg()
    
    model = VO(cfg).to(device)
    model.train() # 학습 모드

    # 2. 더미 데이터 생성
    batch = create_dummy_batch()

    # 3. Forward Pass
    print("▶ Forward 진행 중...")
    output = model(batch, iters=4) # 테스트용으로 4회 반복

    # 4. 출력 값 검증
    poses = output['poses']
    final_pose = output['final_pose']
    
    assert len(poses) == 4, f"Iteration 결과 개수 불일치: {len(poses)}"
    assert isinstance(final_pose, SE3), "최종 포즈가 SE3 객체가 아님"
    print(f"✅ Forward 성공! 최종 포즈 차원: {final_pose.shape}")

    # 5. Backward Pass 테스트 (Gradient Flow 체크)
    print("▶ Backward 및 Gradient Flow 체크 중...")
    # 간단한 Pose Loss (GT와의 차이)
    gt_pose = SE3.InitFromVec(batch['rel_pose'])
    
    # Sequence Loss: 모든 iteration 결과에 대해 로스 계산
    total_loss = 0
    for i, p in enumerate(poses):
        # Geodesic distance on SE3
        diff = (gt_pose.inv() * p).log() # [B, 6]
        total_loss += diff.abs().mean() * (0.8 ** (len(poses) - i - 1))

    total_loss.backward()

    # 6. 각 모듈의 가중치 업데이트 여부 확인
    modules_to_check = {
        "GAT": model.initializer.gat,
        "TriangleHead": model.initializer.tri_head,
        "UpdateBlock (GRU)": model.update_block.gru,
        "Damping (Lambda)": model.log_lmbda
    }

    for name, module in modules_to_check.items():
        if isinstance(module, nn.Parameter):
            grad = module.grad
        else:
            # 첫 번째 파라미터의 기울기 확인
            grad = next(module.parameters()).grad
            
        if grad is not None:
            print(f"✅ {name}: Gradient 전파 확인 (Mean Grad: {grad.abs().mean().item():.6f})")
        else:
            print(f"❌ {name}: Gradient 전파 안 됨!")

    print("\n✨ 모든 테스트가 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    test_vo_forward_backward()