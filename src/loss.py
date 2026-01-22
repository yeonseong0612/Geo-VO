import torch
import torch.nn.functional as F
from lietorch import SE3

def pose_geodesic_loss(pred_poses, gt_pose, gamma=0.8):
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)

    n_iters, B = pred_poses.shape[:2]
    gt_pose_expanded = gt_pose.view(1, B) 
    
    relative_pose = pred_poses.inv() * gt_pose_expanded
    diff = relative_pose.log() # [8, 4, 6] -> (trans_x, y, z, rot_x, y, z)
    
    # --- 비중 조절 구간 ---
    trans_err = diff[..., :3].norm(dim=-1) # 이동 오차
    rot_err = diff[..., 3:].norm(dim=-1)   # 회전 오차
    
    # 보통 회전(rad)이 작으므로 회전에 높은 가중치(w_rot)를 줍니다.
    # 예: 이동 1.0 : 회전 100.0 (데이터셋에 따라 조정)
    w_rot = 10.0 
    err = trans_err + w_rot * rot_err 
    # ----------------------
    
    err = err.mean(dim=-1) # 배치의 평균
    weights = gamma ** torch.arange(n_iters, device=err.device).flip(0)
    return (weights * err).sum()

def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    """
    weight_history: [8, B, 800, 1] - GRU가 예측한 신뢰도
    reproj_errors: [8, B, 800, 2]  - 순환 매칭 오차 (u, v 차이)
    """
    n_iters = weight_history.shape[0]

    # 1. 가중 오차 계산 (L1 대신 Robust한 학습을 위해 Huber나 Squared도 고려 가능)
    # 오차가 클수록 weight가 낮아지도록 유도됨
    weighted_err = (weight_history * reproj_errors.abs()).mean(dim=(1, 2, 3))
    
    # 2. 정규화 항 (Weight가 0으로 수렴하는 것을 방지)
    # -log(w)는 w가 1에 가까워지도록 밀어내는 힘입니다.
    log_weight = torch.log(weight_history + 1e-6)
    reg = -log_weight.mean(dim=(1, 2, 3)) 
    
    weights = gamma ** torch.arange(n_iters, device=weighted_err.device).flip(0)
    
    # 3. 최종 합산
    # 순환 오차는 픽셀 단위이므로 스케일이 큼 -> 0.01 같은 작은 상수로 조절
    loss = (weights * (weighted_err + 0.01 * reg)).sum()
    
    return loss

def total_loss(outputs, gt_pose):
    poses_h, weights_h, errors_h = outputs
    
    l_pose = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)
    
    # l_pose와 l_weight의 스케일을 맞추는 것이 핵심입니다.
    # 초기 학습 시 l_pose가 너무 작다면 0.05를 0.1 정도로 높여보세요.
    t_loss = l_pose + 0.05 * l_weight
    
    return t_loss, l_pose.detach(), l_weight.detach()