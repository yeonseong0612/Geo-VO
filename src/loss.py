import torch
import torch.nn.functional as F
from lietorch import SE3

def pose_geodesic_loss(poses_h, gt_pose, gamma=0.8):
    
    n_iters = poses_h.shape[0]
    B = gt_pose.shape[0]
    
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)
        
    total_loss = 0.0
    total_t_err = 0.0
    total_r_err = 0.0
    
    for i in range(n_iters):
        # i번째 이터레이션 예측값 추출 및 SE3 변환
        pred_data = poses_h[i]
        pred_pose = SE3(pred_data) if isinstance(pred_data, torch.Tensor) else pred_data
        
        # 이제 .inv() 연산이 가능합니다.
        # dP = pred * gt.inv() -> 두 포즈 사이의 차이 계산
        diff = pred_pose * gt_pose.inv()
        
        # Geodesic Error 추출 (Log map)
        v = diff.log() # [B, 6] -> (tx, ty, tz, rx, ry, rz)
        v = torch.clamp(v, min=-10.0, max=10.0)

        t_err = v[:, :3].norm(dim=-1).mean()
        r_err = v[:, 3:].norm(dim=-1).mean()
        
        weight = gamma ** (n_iters - i - 1)

        t_weight = 10.0
        r_weight = 1.0 

        total_loss += weight * (t_weight * t_err + r_weight * r_err)
        
        total_t_err += t_err.item()
        total_r_err += r_err.item()

    return total_loss / n_iters, total_t_err / n_iters, total_r_err / n_iters

def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    n_iters = weight_history.shape[0]
    
    huber_err = F.huber_loss(reproj_errors, torch.zeros_like(reproj_errors), 
                             reduction='none', delta=1.0)
    
    weighted_err = (weight_history * huber_err).mean(dim=(1, 2, 3))
    
    reg = torch.pow(weight_history - 1.0, 2).mean(dim=(1, 2, 3))
    
    i = torch.arange(n_iters, device=weighted_err.device)
    weights = gamma**(n_iters - i - 1)
    
    loss = (weights * (weighted_err + 0.01 * reg)).sum() 
    return loss

def total_loss(outputs, gt_pose):
    poses_h, weights_h, errors_h = outputs
    
    l_pose, t_err, r_err = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)

    t_loss = l_pose + 0.01 * l_weight 
    
    return t_loss, t_err, r_err, l_weight.detach()