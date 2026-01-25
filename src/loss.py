import torch
import torch.nn.functional as F
from lietorch import SE3

def pose_geodesic_loss(poses_h, gt_pose, gamma=0.8):
    """
    [Inputs]
    - poses_h: List of SE3 objects (length: iters)
    - gt_pose: SE3 object or [B, 7] tensor
    """
    n_iters = len(poses_h)
    
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)
        
    total_loss = 0.0
    final_t_err = 0.0
    final_r_err = 0.0
    
    for i in range(n_iters):
        pred_pose = poses_h[i]
        
        # Geodesic distance on SE(3)
        # dP = pred * gt.inv()
        diff = pred_pose * gt_pose.inv()
        v = diff.log() # [B, 6]
        
        t_err = v[:, :3].norm(dim=-1).mean()
        r_err = v[:, 3:].norm(dim=-1).mean()
        
        # Exponential discounting
        weight = gamma ** (n_iters - i - 1)

        # 1m 오차와 약 0.5도 오차를 비슷한 중요도로 설정 (t:1, r:100)
        total_loss += weight * (1.0 * t_err + 100.0 * r_err)
        
        # 마지막 이터레이션의 에러를 기록 (모니터링용)
        if i == n_iters - 1:
            final_t_err = t_err.item()
            final_r_err = r_err.item()

    return total_loss, final_t_err, final_r_err
def weight_reg_loss(weight_history, gamma=0.8):
    n_iters = len(weight_history)
    loss = 0.0
    for i in range(n_iters):
        w = weight_history[i][..., 0:1] # Confidence channel
        
        # 1. 가중치 하한선 강제 (0으로 죽는 것 방지)
        # 가중치가 0.01보다 작아지면 급격하게 로스를 부여합니다.
        # 이를 통해 자코비안이 Singular Matrix가 되는 것을 물리적으로 막습니다.
        floor_loss = torch.relu(0.1 - w).mean() 
        
        # 2. 기존의 1.0 정규화 (편식 억제)
        reg = torch.pow(w - 1.0, 2).mean()
        
        step_weight = gamma**(n_iters - i - 1)
        # 하한선 로스 비중을 높게 설정하여 시스템 안정성을 최우선으로 합니다.
        loss += step_weight * (reg + 10.0 * floor_loss)
        
    return loss

def total_loss(outputs, gt_pose_tensor):
    poses_h = outputs['poses']
    weights_h = outputs['weights']
    
    gt_pose = SE3(gt_pose_tensor)
    l_pose, t_err, r_err = pose_geodesic_loss(poses_h, gt_pose)
    
    # 3. 가중치 로스 계산
    l_weight = weight_reg_loss(weights_h)

    # [핵심 변경] 가중치 로스 계수를 아주 작게(1e-5)라도 주어 
    # 최소한의 수치적 안정성(Epsilon 역할)을 확보합니다.
    t_loss = l_pose + 1e-5 * l_weight 
    
    return t_loss, t_err, r_err, l_weight.detach()