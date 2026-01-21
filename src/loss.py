import torch

from lietorch import SE3

def pose_geodesic_loss(pred_poses, gt_pose, gamma=0.8):

    # 1. 타입 변환
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)

    # 2. 차원 정보 추출
    n_iters = pred_poses.shape[0] # 8
    B = pred_poses.shape[1]      # 4

    # 3. 차원 확장 ([4] -> [1, 4])
    # unsqueeze 대신 view를 사용하며, 반드시 튜플 (1, B)를 전달합니다.
    gt_pose_expanded = gt_pose.view((1, B)) 

    # 4. 상대 포즈 및 Log map 계산
    relative_pose = pred_poses.inv() * gt_pose_expanded
    diff = relative_pose.log() # [8, 4, 6]
    
    # 5. Loss 계산
    err = diff.norm(dim=-1).mean(dim=-1)
    weights = gamma ** torch.arange(n_iters, device=err.device).flip(0)
    loss = (weights * err).sum()
    
    return loss
def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    n_iters = weight_history.shape[0]

    # 1. 가중 오차 계산
    weighted_err = (weight_history * reproj_errors.abs()).mean(dim=(1, 2, 3))
    
    # 2. 정규화 항 계산 (NaN 방지를 위해 clamp 추가)
    # log 안의 값이 너무 작아져서 reg가 폭발하는 것을 방지합니다.
    log_weight = torch.log(weight_history + 1e-6)
    reg = -torch.clamp(log_weight, min=-10.0).mean(dim=(1, 2, 3)) 
    
    weights = gamma ** torch.arange(n_iters, device=weighted_err.device).flip(0)
    
    # 0.01 * reg로 비중을 낮춰서 합산
    loss = (weights * (weighted_err + 0.01 * reg)).sum()
    
    return loss

def total_loss(outputs, gt_pose):
    poses_h, weights_h, errors_h = outputs
    
    l_pose = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)
    
    t_loss = l_pose + 0.01 * l_weight
    
    return t_loss, l_pose.detach(), l_weight.detach()