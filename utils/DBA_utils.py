import torch

def compute_projection_jacobian(kpts_2d, depth, intrinsics):
    # kpts_2d: [B, N, 2], depth: [B, N, 1], intrinsics: [B, 4]
    B, N, _ = kpts_2d.shape
    device = kpts_2d.device
    
    # 배치를 고려한 intrinsics 분리
    fx = intrinsics[:, 0:1] # [B, 1]
    fy = intrinsics[:, 1:2]
    cx = intrinsics[:, 2:3]
    cy = intrinsics[:, 3:4]

    # 3D 포인트 복원 (Back-projection)
    u, v = kpts_2d[:, :, 0:1], kpts_2d[:, :, 1:2]
    Z = depth.clamp(min=0.1)
    
    X = (u - cx.unsqueeze(1)) * Z / fx.unsqueeze(1)
    Y = (v - cy.unsqueeze(1)) * Z / fy.unsqueeze(1)
    
    inv_Z = 1.0 / Z
    inv_Z2 = inv_Z * inv_Z

    # --- J_p [B, N, 2, 6] ---
    J_p = torch.zeros((B, N, 2, 6), device=device)
    
    # Translation (tx, ty, tz)
    J_p[..., 0, 0] = (fx.unsqueeze(1) * inv_Z).squeeze(-1)
    J_p[..., 0, 1] = 0
    J_p[..., 0, 2] = (-fx.unsqueeze(1) * X * inv_Z2).squeeze(-1)
    
    J_p[..., 1, 0] = 0
    J_p[..., 1, 1] = (fy.unsqueeze(1) * inv_Z).squeeze(-1)
    J_p[..., 1, 2] = (-fy.unsqueeze(1) * Y * inv_Z2).squeeze(-1)
    
    # Rotation (rx, ry, rz) - 표준 수식 적용
    J_p[..., 0, 3] = (-fx.unsqueeze(1) * X * Y * inv_Z2).squeeze(-1)
    J_p[..., 0, 4] = (fx.unsqueeze(1) * (1 + (X**2 * inv_Z2))).squeeze(-1)
    J_p[..., 0, 5] = (-fx.unsqueeze(1) * Y * inv_Z).squeeze(-1)
    
    J_p[..., 1, 3] = (-fy.unsqueeze(1) * (1 + (Y**2 * inv_Z2))).squeeze(-1)
    J_p[..., 1, 4] = (fy.unsqueeze(1) * X * Y * inv_Z2).squeeze(-1)
    J_p[..., 1, 5] = (fy.unsqueeze(1) * X * inv_Z).squeeze(-1)


    J_d = J_p[..., 2:3].clone() 

    return J_p, J_d