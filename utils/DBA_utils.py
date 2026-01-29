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

def compute_geometry_data(pts_3d, kpts_target, pose, depth, intrinsics):
    """
    pts_3d: [B, N, 3] (현재 프레임의 3D 좌표)
    kpts_target: [B, N, 2] (타겟 프레임의 관측된 2D 특징점 좌표)
    pose: [B, 1] (lietorch SE3 객체, 확장된 상태)
    depth: [B, N, 1] (현재 추정된 깊이)
    intrinsics: [B, 4] (fx, fy, cx, cy)
    """
    B, N, _ = pts_3d.shape
    device = pts_3d.device

    # 1. 3D 포인트 변환 (Current -> Target)
    # pose: [B, 1, 7], pts_3d: [B, N, 3] -> 결과: [B, N, 3]
    pts_tp1_pred_3d = pose * pts_3d 

    # 2. 2D 투영 (Perspective Projection)
    fx, fy, cx, cy = intrinsics[:, 0], intrinsics[:, 1], intrinsics[:, 2], intrinsics[:, 3]
    
    # 차원 맞추기
    fx = fx.view(B, 1); fy = fy.view(B, 1); cx = cx.view(B, 1); cy = cy.view(B, 1)
    
    X, Y, Z = pts_tp1_pred_3d[..., 0:1], pts_tp1_pred_3d[..., 1:2], pts_tp1_pred_3d[..., 2:3]
    inv_Z = 1.0 / Z.clamp(min=0.1)

    u_pred = fx * X * inv_Z + cx
    v_pred = fy * Y * inv_Z + cy
    kpts_pred = torch.cat([u_pred, v_pred], dim=-1) # [B, N, 2]

    # 3. Residual 계산 (Observed - Predicted)
    r = kpts_target - kpts_pred # [B, N, 2]

    # 4. Jacobian 계산
    J_p, J_d = compute_projection_jacobian(kpts_pred, Z, intrinsics)

    return r, J_p, J_d