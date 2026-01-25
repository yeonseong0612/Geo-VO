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
        total_loss += weight * (1.0 * t_err + 50.0 * r_err)
        
        # 마지막 이터레이션의 에러를 기록 (모니터링용)
        if i == n_iters - 1:
            final_t_err = t_err.item()
            final_r_err = r_err.item()

    return total_loss, final_t_err, final_r_err

def weight_reg_loss(weight_history, reproj_errors=None, gamma=0.8):
    # 아직 reproj_errors 연동 전이라면 간단한 Regularization만 수행
    n_iters = len(weight_history)
    loss = 0.0
    for i in range(n_iters):
        w = weight_history[i][..., 0:1] # Confidence channel
        reg = torch.pow(w - 1.0, 2).mean()
        
        step_weight = gamma**(n_iters - i - 1)
        loss += step_weight * reg
        
    return loss

def total_loss(outputs, gt_pose_tensor):
    """
    [Inputs]
    - outputs: VO 모델의 리턴 딕셔너리
    - gt_pose_tensor: [B, 7] 형태의 상대 포즈 정답
    """
    poses_h = outputs['poses']
    weights_h = outputs['weights']
    
    # 1. SE3 객체로 변환
    gt_pose = SE3(gt_pose_tensor)
    
    # 2. 포즈 지오데식 로스 계산
    l_pose, t_err, r_err = pose_geodesic_loss(poses_h, gt_pose)
    
    # 3. 가중치 정규화 로스 (가중치가 0으로 죽는 것 방지)
    l_weight = weight_reg_loss(weights_h)

    # 최종 로스 조합 (가중치 로스는 작게 시작)
    t_loss = l_pose + 0.01 * l_weight 
    
    # train.py에서 4개의 인자를 받으므로 맞춰서 반환
    return t_loss, t_err, r_err, l_weight.detach()  