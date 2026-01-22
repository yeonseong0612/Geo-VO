import torch
import torch.nn.functional as F
from lietorch import SE3

def pose_geodesic_loss(pred_poses, gt_pose, gamma=0.8):
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)

    n_iters, B = pred_poses.shape[:2]
    try:
        gt_pose_expanded = gt_pose[None]
    except:
        gt_pose_expanded = SE3(gt_pose.data.view(1, B, 7))
    
    relative_pose = pred_poses.inv() * gt_pose_expanded
    diff = relative_pose.log()
    
    trans_err = diff[..., :3].norm(dim=-1)
    rot_err = diff[..., 3:].norm(dim=-1)
    
    w_rot = 10.0 
    err = trans_err + w_rot * rot_err 
    
    i = torch.arange(n_iters, device=err.device)
    weights = gamma**(n_iters - i - 1)
    
    loss = (weights * err.mean(dim=-1)).sum()
    return loss

def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    n_iters = weight_history.shape[0]

    # --- [수정 1] 가중치 0 수렴 방지 (Clamp) ---
    # 모델이 0을 내뱉어도 최소 0.01은 유지하게 하여 꼼수를 차단합니다.
    weight_history = torch.clamp(weight_history, min=0.01)

    # --- [수정 2] 휴버 로스 델타 하향 (제안하신 내용) ---
    # 0.5 픽셀만 틀려도 모델이 오차를 민감하게 느끼게 합니다.
    huber_err = F.huber_loss(reproj_errors, torch.zeros_like(reproj_errors), 
                             reduction='none', delta=0.5)
    
    weighted_err = (weight_history * huber_err).mean(dim=(1, 2, 3))
    
    # --- [수정 3] 정규화 계수 강화 ---
    # 가중치를 키우라는 압박(reg)을 더 세게 줍니다.
    log_weight = torch.log(weight_history) 
    reg = -log_weight.mean(dim=(1, 2, 3)) 
    
    weights = gamma ** torch.arange(n_iters, device=weighted_err.device).flip(0)
    
    # reg 앞에 1.0(혹은 그 이상)을 곱해 가중치 유지를 강요합니다.
    loss = (weights * (weighted_err + 1.0 * reg)).sum() 
    return loss

def total_loss(outputs, gt_pose):
    poses_h, weights_h, errors_h = outputs
    
    l_pose = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)

    # --- [수정 4] 전체 밸런스 조정 ---
    # Pose Loss가 움직이게 하려면 Weight Loss도 무시 못 할 수준으로 더해줘야 합니다.
    t_loss = l_pose + 1.0 * l_weight 
    
    return t_loss, l_pose.detach(), l_weight.detach()