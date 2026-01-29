import torch

def compute_projection_jacobian(kpts_2d, depth, intrinsics):
    B, N, _ = kpts_2d.shape
    device = kpts_2d.device
    
    # [B, 1] 형태로 추출 (B, 1, 1보다 [B, 1]이 연산 시 더 안전합니다)
    fx = intrinsics[:, 0].view(B, 1)
    fy = intrinsics[:, 1].view(B, 1)
    cx = intrinsics[:, 2].view(B, 1)
    cy = intrinsics[:, 3].view(B, 1)

    # u, v, Z를 [B, N]으로 펼침
    u = kpts_2d[..., 0] # [B, N]
    v = kpts_2d[..., 1] # [B, N]
    Z = depth.view(B, N).clamp(min=0.1) # [B, N] 확실히 고정
    inv_Z = 1.0 / Z

    # 1. 정규화 좌표계 (Normalized Coordinates) - [B, N]
    x = (u - cx) / fx
    y = (v - cy) / fy

    # J_p 초기화: [B, N, 2, 6]
    J_p = torch.zeros((B, N, 2, 6), device=device)
    
    # 2. 값 할당 (우변 연산 결과는 모두 [B, N]이 되어야 함)
    # --- Translation (tx, ty, tz) ---
    J_p[:, :, 0, 0] = fx * inv_Z       # [B, N]
    J_p[:, :, 1, 1] = fy * inv_Z       # [B, N]
    
    J_p[:, :, 0, 2] = -fx * x * inv_Z  # [B, N]
    J_p[:, :, 1, 2] = -fy * y * inv_Z  # [B, N]
    
    # --- Rotation (rx, ry, rz) ---
    J_p[:, :, 0, 3] = -fx * x * y      # [B, N]
    J_p[:, :, 1, 3] = -fy * (1 + y**2) # [B, N]
    
    J_p[:, :, 0, 4] = fx * (1 + x**2)  # [B, N]
    J_p[:, :, 1, 4] = fy * x * y       # [B, N]
    
    J_p[:, :, 0, 5] = -fx * y          # [B, N]
    J_p[:, :, 1, 5] = fy * x           # [B, N]

    # --- Depth Jacobian (J_d) ---
    # J_d: [B, N, 2, 1]
    # d_uv / d_depth 연산 (일반적으로 -fx*X/Z^2 로 계산)
    J_d = torch.zeros((B, N, 2, 1), device=device)
    J_d[:, :, 0, 0] = -fx * x * inv_Z
    J_d[:, :, 1, 0] = -fy * y * inv_Z

    return J_p, J_d

def compute_geometry_data(pts_3d, kpts_target, pose, depth, intrinsics):
    B, N, _ = pts_3d.shape
    
    # 1. 3D 포인트 변환 (명시적 transform 호출)
    # pose가 [B, 1] 형태의 SE3 객체라면 .act()를 사용해 포인트를 변환합니다.
    # pts_3d: [B, N, 3] -> pts_tp1_pred_3d: [B, N, 3]
    pts_tp1_pred_3d = pose.act(pts_3d) 

    # 2. 2D 투영 파라미터 준비
    # view(B, 1)은 아주 좋습니다.
    fx = intrinsics[:, 0].view(B, 1)
    fy = intrinsics[:, 1].view(B, 1)
    cx = intrinsics[:, 2].view(B, 1)
    cy = intrinsics[:, 3].view(B, 1)
    
    # 여기서 각 성분의 차원을 [B, N]으로 확실히 분리합니다.
    X = pts_tp1_pred_3d[..., 0] # [B, N]
    Y = pts_tp1_pred_3d[..., 1] # [B, N]
    Z = pts_tp1_pred_3d[..., 2] # [B, N]
    
    inv_Z = 1.0 / Z.clamp(min=0.1) # [B, N]

    # 이제 [B, 1] * [B, N] * [B, N] + [B, 1] 연산이 수행됩니다.
    u_pred = fx * X * inv_Z + cx # [B, N]
    v_pred = fy * Y * inv_Z + cy # [B, N]
    
    # unsqueeze로 마지막 차원을 살려줍니다.
    kpts_pred = torch.stack([u_pred, v_pred], dim=-1) # [B, N, 2]

    # 3. Residual
    r = kpts_target - kpts_pred 

    # 4. Jacobian (Z의 차원 확인: [B, N]이어야 함)
    J_p, J_d = compute_projection_jacobian(kpts_pred, Z, intrinsics)

    return r, J_p, J_d  