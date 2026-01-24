import torch

def compute_projection_jacobian(kpts_2d, depth, intrinsics):
    B, N, _ = kpts_2d.shape
    device = kpts_2d.device
    
    fx = intrinsics[:, 0].view(B, 1, 1)
    fy = intrinsics[:, 1].view(B, 1, 1)
    cx = intrinsics[:, 2].view(B, 1, 1)
    cy = intrinsics[:, 3].view(B, 1, 1)

    u, v = kpts_2d[..., 0:1], kpts_2d[..., 1:2]
    Z = depth.clamp(min=0.1)
    inv_Z = 1.0 / Z

    # 1. 정규화 좌표계 (Normalized Coordinates)
    x = (u - cx) / fx
    y = (v - cy) / fy

    J_p = torch.zeros((B, N, 2, 6), device=device)
    
    # --- Translation (tx, ty, tz) ---
    # lietorch SE3.exp는 tangent space에서의 변화를 다룹니다.
    J_p[:, :, 0, 0] = (fx * inv_Z).squeeze(-1)
    J_p[:, :, 1, 1] = (fy * inv_Z).squeeze(-1)
    
    # 핵심 수정: tz에 의한 변화량 (u-cx)/Z와 부호 일치 확인
    J_p[:, :, 0, 2] = (-fx * x * inv_Z).squeeze(-1)
    J_p[:, :, 1, 2] = (-fy * y * inv_Z).squeeze(-1)
    
    # --- Rotation (rx, ry, rz) ---
    J_p[:, :, 0, 3] = (-fx * x * y).squeeze(-1)
    J_p[:, :, 1, 3] = (-fy * (1 + y**2)).squeeze(-1)
    
    J_p[:, :, 0, 4] = (fx * (1 + x**2)).squeeze(-1)
    J_p[:, :, 1, 4] = (fy * x * y).squeeze(-1)
    
    J_p[:, :, 0, 5] = (-fx * y).squeeze(-1) 
    J_p[:, :, 1, 5] = (fy * x).squeeze(-1)

    J_d = J_p[:, :, :, 2:3].clone() 

    return J_p, J_d