import torch
import torch.nn.functional as F
from lietorch import SE3

def matrix_to_7vec_batch(matrix):
    """[B, 4, 4] -> [B, 7] (x, y, z, qx, qy, qz, qw)"""
    B = matrix.shape[0]
    t = matrix[:, :3, 3]
    from utils.geo_utils import matrix_to_quat 
    q = matrix_to_quat(matrix[:, :3, :3]) 
    return torch.cat([t, q], dim=-1)

def pose_geodesic_loss(poses_h, gt_pose, residuals_h, gamma=0.8):
    """
    poses_h: 모델의 예측 포즈 리스트
    gt_pose: 정답 포즈 (SE3)
    residuals_h: 모델의 forward에서 계산된 재투영 잔차 [B, iters, N, 2]
    """
    n_iters = len(poses_h)
    total_loss = 0.0
    t_err_mon, r_err_mon = 0.0, 0.0
    
    gt_t_vec = gt_pose.vec()[:, :3]
    dist_gt = torch.norm(gt_t_vec.detach(), dim=-1).clamp(min=0.5) 
    
    for i in range(n_iters):
        pred_pose_mat = poses_h[i]
        pred_pose = SE3.InitFromVec(matrix_to_7vec_batch(pred_pose_mat))
        
        # 1. Pose Loss (Geodesic)
        relative_diff = gt_pose.inv() * pred_pose
        log_map = relative_diff.log() 

        # Translation Loss
        t_err_vec = log_map[:, :3]
        t_err_norm = torch.sqrt(torch.sum(t_err_vec**2, dim=-1) + 1e-8)
        loss_t = (t_err_norm / (dist_gt + 1e-7)).mean()

        # Rotation Loss
        r_err_vec = log_map[:, 3:] 
        r_err_norm = torch.sqrt(torch.sum(r_err_vec**2, dim=-1) + 1e-8)
        loss_r = r_err_norm.mean()      

        # 2. [신규] Reprojection Loss (Joint Error)
        r = residuals_h[i] 
        loss_reproj = torch.sqrt(torch.sum(r**2, dim=-1) + 1e-8).mean()

        step_weight = gamma ** (n_iters - i - 1)
        

        step_loss = (2.0 * loss_t) + (5.0 * loss_r) + (1.0 * loss_reproj)
        
        total_loss += step_weight * torch.clamp(step_loss, max=100.0)
        
        if i == n_iters - 1:
            t_err_mon = t_err_norm.detach().mean().item()
            r_err_mon = loss_r.detach().item()

    return total_loss, t_err_mon, r_err_mon

def weight_reg_loss(weight_history, gamma=0.8):
    n_iters = len(weight_history)
    loss = 0.0
    for i in range(n_iters):
        w = weight_history[i]
        floor_loss = torch.relu(0.05 - w).mean() 
        reg = torch.pow(w - 1.0, 2).mean() 
        step_weight = gamma**(n_iters - i - 1)
        loss += step_weight * (reg + 5.0 * floor_loss)
    return loss

def total_loss(outputs, batch):
    # NaN 방지용 모니터링
    for k, v in outputs.items():
        if isinstance(v, list) and any(torch.isnan(t).any() for t in v):
            print(f"⚠️ NaN in list {k}")

    device = outputs['pose_matrices'][0].device
    gt_pose = SE3(batch['rel_pose'].to(device))
    
    # forward에서 뱉은 residuals를 같이 넘겨줍니다.
    # 만약 outputs에 'residuals'가 없다면 forward의 return에 추가해야 합니다.
    l_pose, t_err, r_err = pose_geodesic_loss(
        outputs['pose_matrices'], 
        gt_pose, 
        outputs['residuals']
    )
    
    l_weight = weight_reg_loss(outputs['confidences'])

    # 최종 손실
    final_loss = l_pose + 0.01 * l_weight 
    
    return final_loss, t_err, r_err, l_weight.detach()