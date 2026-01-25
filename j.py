import torch
import torch.nn as nn
from utils.DBA_utils import compute_projection_jacobian
from src.model import VO
from CFG.vo_cfg import vo_cfg
from lietorch import SE3

# nn.Module의 기본 기능은 유지하되, 무거운 로드는 피하는 Mock 클래스
class MockVO(VO):
    def __init__(self):
        # nn.Module의 필수 내부 변수들을 초기화 (이걸 해야 .to(device)가 작동함)
        nn.Module.__init__(self) 
        # 부모 클래스(VO)의 __init__은 호출하지 않음으로써 SuperPoint 로드 회피

def test_jacobian_convergence():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 1, 800
    
    # 1. Mock 객체 생성 및 장치 이동
    vo_tester = MockVO().to(device)
    
    # 2. 데이터 준비 (이하 동일)
    intrinsics = torch.tensor([[600.0, 600.0, 400.0, 300.0]], device=device)
    kpts = torch.randn(B, N, 2, device=device) * 50 + 400
    depth = torch.ones(B, N, 1, device=device) * 10.0
    
    # 3. 정답 포즈(Target) 설정
    target_vec = torch.tensor([[0.05, -0.03, 0.1, 0.01, -0.01, 0.005]], device=device)
    target_pose = SE3.exp(target_vec)
    
    # 정답 픽셀 위치
    p_target = vo_tester.projector(kpts, depth, target_pose, intrinsics)
    
    # 4. 현재 포즈 초기화
    cur_pose = SE3.Identity(B, device=device)
    
    print("\n" + "="*60)
    print("🚀 자코비안 기반 포즈 수렴 테스트 (Gauss-Newton)")
    print("="*60)

    for i in range(15):
        p_cur = vo_tester.projector(kpts, depth, cur_pose, intrinsics)
        
        # Residual 계산
        residual = p_target - p_cur 
        mse = residual.pow(2).mean().item()
        
        # 자코비안 계산
        J, _ = compute_projection_jacobian(kpts, depth, intrinsics)
        
        # Gauss-Newton 시스템 구성 (H = J^T * J, g = J^T * r)
        J_t = J.transpose(-1, -2)
        H = torch.matmul(J_t, J)
        g = torch.matmul(J_t, residual.unsqueeze(-1))
        
        H_sum = H.sum(dim=1) + torch.eye(6, device=device) * 1e-4 # Damping
        g_sum = g.sum(dim=1)
        
        # 업데이트량 계산 (H * delta = g)
        delta = torch.linalg.solve(H_sum, g_sum).squeeze(-1)
        
        # 포즈 업데이트
        cur_pose = SE3.exp(delta) * cur_pose
        
        print(f"Iter {i+1:02d} | MSE: {mse:12.6f} | Delta Norm: {delta.norm().item():.6f}")
        
        if mse < 1e-6: break

    print("="*60)
    if mse < 1e-4:
        print("✅ 결과: 성공! 자코비안이 포즈를 정답으로 정확히 유도합니다.")
    else:
        print("❌ 결과: 실패! 수렴하지 않습니다. (부호나 수식을 점검하세요)")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_jacobian_convergence()