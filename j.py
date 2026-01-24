import torch
import torch.nn as nn
from utils.DBA_utils import compute_projection_jacobian
from src.model import VO
from CFG.vo_cfg import vo_cfg
from lietorch import SE3

def test_jacobian_precision():
    # 1. 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 2, 800
    eps = 1e-3  # 미세 변화량 (너무 작으면 float32 정밀도 문제, 너무 크면 선형 근사 문제)
    
    # 2. 가상의 입력 데이터 (현실적인 주행 상황 가정)
    # 이미지 중앙 근처의 키포인트들
    kpts = torch.randn(B, N, 2, device=device) * 50 + 400 
    # 10미터 앞의 평면
    depth = torch.ones(B, N, 1, device=device) * 10.0
    # 표준적인 카메라 파라미터
    intrinsics = torch.tensor([[600.0, 600.0, 400.0, 300.0]], device=device).repeat(B, 1)
    
    # 3. 분석적 자코비안 계산 (우리가 만든 함수)
    J_p, _ = compute_projection_jacobian(kpts, depth, intrinsics)
    
    # 검증할 파라미터 선택: Translation Z (tz)
    # J_p의 인덱스: 0:tx, 1:ty, 2:tz, 3:rx, 4:ry, 5:rz
    ana_grad_tz = J_p[..., 2] 

    # 4. 수치적 미분 계산 (Central Difference 방식)
    # f'(x) ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
    vo_tester = VO(vo_cfg).to(device)
    identity = SE3.Identity(B, device=device)

    # (A) f(x + eps) 계산
    delta_plus = torch.zeros(B, 6, device=device)
    delta_plus[:, 2] = eps # tz 방향으로 +eps
    p_plus = vo_tester.projector(kpts, depth, SE3.exp(delta_plus) * identity, intrinsics)

    # (B) f(x - eps) 계산
    delta_minus = torch.zeros(B, 6, device=device)
    delta_minus[:, 2] = -eps # tz 방향으로 -eps
    p_minus = vo_tester.projector(kpts, depth, SE3.exp(delta_minus) * identity, intrinsics)

    # (C) 수치적 기울기 산출
    num_grad_tz = (p_plus - p_minus) / (2 * eps)

    # 5. 결과 비교 및 출력
    diff = torch.abs(num_grad_tz - ana_grad_tz)
    mean_error = diff.mean().item()
    max_error = diff.max().item()

    print("\n" + "="*60)
    print("🔍 자코비안 정밀 검증 결과 (Central Difference)")
    print("="*60)
    print(f"Numerical Gradient (tz) Mean:  {num_grad_tz.mean().item():.8f}")
    print(f"Analytical Gradient (tz) Mean: {ana_grad_tz.mean().item():.8f}")
    print("-" * 60)
    print(f"Mean Absolute Error (MAE):     {mean_error:.10f}")
    print(f"Max Absolute Error:             {max_error:.10f}")
    print("="*60)

    if mean_error < 1e-3:
        print("✅ 결과: 자코비안 수식이 수치적으로 완벽하게 일치합니다!")
    else:
        print("⚠️ 주의: 오차가 여전합니다. 수식의 부호나 단위를 재점검하세요.")
    print("="*60 + "\n")

if __name__ == "__main__":
    test_jacobian_precision()