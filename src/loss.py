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
    """
    weight_history: [Iter, B, N, 2]
    reproj_errors: [Iter, B, N, 2]
    """
    n_iters = weight_history.shape[0]

    # 1. 모든 배치(B)와 모든 노드(N), 그리고 좌표축(2)에 대해 평균을 냅니다.
    # [Iter, B, N, 2] -> [Iter]
    weighted_err = (weight_history * reproj_errors.abs()).mean(dim=(1, 2, 3))
    
    # 2. 정규화 항도 동일하게 차원을 축소합니다.
    # [Iter, B, N, 2] -> [Iter]
    reg = -torch.log(weight_history + 1e-6).mean(dim=(1, 2, 3))
    
    # 3. 가중치 텐서 생성 [Iter]
    weights = gamma ** torch.arange(n_iters, device=weighted_err.device).flip(0)
    
    # 4. 이제 [8] * ([8] + [8]) 연산이므로 정상적으로 수행됩니다.
    loss = (weights * (weighted_err + 0.01 * reg)).sum()
    
    return loss

def total_loss(outputs, gt_pose):
    poses_h, weights_h, errors_h = outputs
    
    l_pose = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)
    
    t_loss = l_pose + 0.1 * l_weight
    
    return t_loss, l_pose.detach(), l_weight.detach()