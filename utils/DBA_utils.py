import torch

def compute_projection_jacobian(kpts_2d, depth, intrinsics):
    # 1. 차원 맞추기
    if kpts_2d.dim() == 2: kpts_2d = kpts_2d.unsqueeze(0)
    if depth.dim() == 2: depth = depth.unsqueeze(0)
    
    # intrinsics에서 값 추출 (에러 방지 핵심)
    if intrinsics.dim() == 2:
        intrinsics_vals = intrinsics[0]
    else:
        intrinsics_vals = intrinsics
    fx, fy, cx, cy = intrinsics_vals[0], intrinsics_vals[1], intrinsics_vals[2], intrinsics_vals[3]

    B, N, _ = kpts_2d.shape
    device = kpts_2d.device

    # 2. 3D 점 복원
    u, v = kpts_2d[:, :, 0:1], kpts_2d[:, :, 1:2]
    Z = depth + 1e-6 # Divide by zero 방지
    
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    
    inv_Z = 1.0 / Z
    inv_Z2 = inv_Z * inv_Z

    # 연산을 위해 Squeeze (B, N)
    X_s, Y_s, inv_Z_s, inv_Z2_s = X.squeeze(-1), Y.squeeze(-1), inv_Z.squeeze(-1), inv_Z2.squeeze(-1)

    # 3. J_p [B, N, 2, 6] (Pose Jacobian: translation + rotation)
    J_p = torch.zeros((B, N, 2, 6), device=device, dtype=kpts_2d.dtype)
    
    # Translation (tx, ty, tz)
    J_p[:, :, 0, 0] = fx * inv_Z_s
    J_p[:, :, 0, 2] = -fx * X_s * inv_Z2_s
    J_p[:, :, 1, 1] = fy * inv_Z_s
    J_p[:, :, 1, 2] = -fy * Y_s * inv_Z2_s
    
    # Rotation (rx, ry, rz)
    J_p[:, :, 0, 3] = -fx * X_s * Y_s * inv_Z2_s
    J_p[:, :, 0, 4] = fx * (1 + (X_s * inv_Z_s)**2)
    J_p[:, :, 0, 5] = -fx * Y_s * inv_Z_s
    J_p[:, :, 1, 3] = -fy * (1 + (Y_s * inv_Z_s)**2)
    J_p[:, :, 1, 4] = fy * X_s * Y_s * inv_Z2_s
    J_p[:, :, 1, 5] = fy * X_s * inv_Z_s

    # 4. J_d [B, N, 2, 1] (Depth Jacobian)
    J_d = torch.zeros((B, N, 2, 1), device=device, dtype=kpts_2d.dtype)
    J_d[:, :, 0, 0] = -fx * X_s * inv_Z2_s
    J_d[:, :, 1, 0] = -fy * Y_s * inv_Z2_s

    return J_p, J_d