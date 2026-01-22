import torch
import torch.nn.functional as F
from lietorch import SE3

def pose_geodesic_loss(pred_poses, gt_pose, gamma=0.8):
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)

    n_iters, B = pred_poses.shape[:2]
    gt_pose_expanded = gt_pose[None].expand(n_iters, B, -1)
    
    relative_pose = pred_poses.inv() * gt_pose_expanded
    diff = relative_pose.log() # [iters, B, 6]
    
    trans_err = diff[..., :3].norm(dim=-1) # [iters, B]
    rot_err = diff[..., 3:].norm(dim=-1)   # [iters, B]
    
    w_rot = 15.0 
    w_trans = 1.0
    
    err = w_trans * trans_err + w_rot * rot_err 
    
    i = torch.arange(n_iters, device=err.device)
    weights = gamma**(n_iters - i - 1)
    
    loss = (weights * err.mean(dim=-1)).sum()
    
    # 모니터링을 위해 분리된 에러 평균값 반환
    return loss, trans_err.mean().detach(), rot_err.mean().detach()

def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    n_iters = weight_history.shape[0]
    
    huber_err = F.huber_loss(reproj_errors, torch.zeros_like(reproj_errors), 
                             reduction='none', delta=0.5)
    
    weighted_err = (weight_history * huber_err).mean(dim=(1, 2, 3))
    
    reg = torch.pow(weight_history - 1.0, 2).mean(dim=(1, 2, 3))
    
    i = torch.arange(n_iters, device=weighted_err.device)
    weights = gamma**(n_iters - i - 1)
    
    loss = (weights * (weighted_err + 0.1 * reg)).sum() 
    return loss

def total_loss(outputs, gt_pose):
    poses_h, weights_h, errors_h = outputs
    
    l_pose, t_err, r_err = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)

    t_loss = l_pose + 0.5 * l_weight 
    
    return t_loss, t_err, r_err, l_weight.detach()