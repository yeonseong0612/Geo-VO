import torch
import torch.nn.functional as F
from lietorch import SE3

def pose_geodesic_loss(poses_h, gt_pose, gamma=0.8):
    """
    poses_h: List of [B, 4, 4] matrices or SE3 objects
    gt_pose: SE3 object [B]
    """
    n_iters = len(poses_h)
    total_loss = 0.0
    final_t_err = 0.0
    final_r_err = 0.0
    
    for i in range(n_iters):
        # 만약 matrix 형태([B, 4, 4])로 들어온다면 SE3로 변환
        pred_pose = poses_h[i]
        if isinstance(pred_pose, torch.Tensor) and pred_pose.shape[-2:] == (4, 4):
            # matrix_to_7vec 유틸리티를 쓰거나 직접 변환
            from utils.geo_utils import matrix_to_quat
            q = matrix_to_quat(pred_pose[:, :3, :3])
            t = pred_pose[:, :3, 3]
            pred_pose = SE3(torch.cat([t, q], dim=-1))
        
        # Geodesic distance: ln(T_gt^-1 * T_pred)
        diff = gt_pose.inv() * pred_pose
        v = diff.log() # [B, 6] (v, omega)
        
        t_err = v[:, :3].norm(dim=-1).mean()
        r_err = v[:, 3:].norm(dim=-1).mean()
        
        weight = gamma ** (n_iters - i - 1)
        # Translation(m) vs Rotation(rad) 스케일 밸런싱
        total_loss += weight * (1.0 * t_err + 100.0 * r_err)
        
        if i == n_iters - 1:
            final_t_err = t_err.item()
            final_r_err = r_err.item()

    return total_loss, final_t_err, final_r_err

def weight_reg_loss(weight_history, gamma=0.8):
    n_iters = len(weight_history)
    loss = 0.0
    for i in range(n_iters):
        w = weight_history[i] # [B, N, 1]
        
        # 가중치가 너무 작아져서 Solver가 NaN을 뱉는 것을 방지 (System Stability)
        floor_loss = torch.relu(0.1 - w).mean() 
        # 가중치가 1 근처에 머물도록 유도 (편식 방지)
        reg = torch.pow(w - 1.0, 2).mean()
        
        step_weight = gamma**(n_iters - i - 1)
        loss += step_weight * (reg + 10.0 * floor_loss)
        
    return loss

def total_loss(outputs, batch):
    """
    outputs: {'pose_matrices': [...], 'confidences': [...]}
    batch: {'rel_pose': [B, 7]}
    """
    # 1. Pose Loss
    gt_pose = SE3(batch['rel_pose'])
    # VO 모델이 뱉는 pose_matrices는 [B, 4, 4] 리스트라고 가정
    l_pose, t_err, r_err = pose_geodesic_loss(outputs['pose_matrices'], gt_pose)
    
    # 2. Weight Regularization
    l_weight = weight_reg_loss(outputs['confidences'])

    # 최종 손실 합산 (l_weight는 수치적 안정성을 위한 보조 장치)
    t_loss = l_pose + 1e-4 * l_weight 
    
    return t_loss, t_err, r_err, l_weight.detach()