import torch
import torch.nn.functional as F
from lietorch import SE3

def pose_geodesic_loss(poses_h, gt_pose, gamma=0.8):
    # poses_h: List of SE3 objects or [iters, B, 7]
    n_iters = len(poses_h)
    B = gt_pose.shape[0]
    
    if isinstance(gt_pose, torch.Tensor):
        gt_pose = SE3(gt_pose)
        
    total_loss = 0.0
    total_t_err = 0.0
    total_r_err = 0.0
    
    for i in range(n_iters):
        pred_pose = poses_h[i] # 이미 SE3 객체일 확률이 높음
        
        # dP = pred * gt.inv() (단, gt_pose가 World-to-Camera 기준일 때)
        diff = pred_pose * gt_pose.inv()
        
        # Log map을 통한 tangent vector 추출 [B, 6]
        v = diff.log() 
        
        # 평행이동 오차 (meters) 및 회전 오차 (radians)
        t_err = v[:, :3].norm(dim=-1).mean()
        r_err = v[:, 3:].norm(dim=-1).mean()
        
        # 지수 가중치 (나중 이터레이션일수록 가중치 높음)
        weight = gamma ** (n_iters - i - 1)

        # 수치적 밸런스 조정: 회전에 100~500 정도의 가중치를 주는 것이 일반적
        # 1m 오차와 0.1rad(약 5.7도) 오차를 비슷하게 취급
        t_weight = 1.0
        r_weight = 100.0 

        total_loss += weight * (t_weight * t_err + r_weight * r_err)
        
        # 모니터링용 (마지막 이터레이션 기준이 더 의미 있음)
        if i == n_iters - 1:
            total_t_err = t_err.item()
            total_r_err = r_err.item()

    return total_loss, total_t_err, total_r_err

def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    # weight_history: [iters, B, N, 1 or 2]
    # reproj_errors: [iters, B, N, 2]
    n_iters = len(weight_history)
    
    loss = 0.0
    for i in range(n_iters):
        w = weight_history[i] # [B, N, 2] 또는 [B, N, 1]
        err = reproj_errors[i] # [B, N, 2]
        
        # 2채널(Confidence, Damping)인 경우 1번 채널(Confidence)만 사용
        conf = w[..., 0:1] 
        
        # Huber loss와 Confidence의 결합: Confidence가 높을 때 오차가 작아야 함
        # -log(conf) 또는 (1-conf) 등을 활용하여 conf가 0이 되는 것을 방지
        huber_err = F.huber_loss(err, torch.zeros_like(err), reduction='none', delta=1.0)
        
        # 핵심: 가중치가 0으로 죽는 것을 막기 위해 -log_prob 형태나 강력한 Reg 추가
        weighted_err = (conf * huber_err).mean()
        
        # 가중치가 1에 가깝게 유지되도록 하는 정규화 항 (강도 조절 필요)
        reg = torch.pow(conf - 1.0, 2).mean()
        
        step_weight = gamma**(n_iters - i - 1)
        loss += step_weight * (weighted_err + 0.1 * reg) # reg 가중치 상향
        
    return loss

def compute_total_loss(outputs, batch):
    # VO 모델의 반환 구조에 맞춤
    poses_h = outputs['poses']
    depths_h = outputs['depths']
    weights_h = outputs['weights']
    
    # GT 데이터 (batch에서 추출)
    gt_pose = batch['gt_pose'].to(poses_h[0].data.device)
    
    # 1. 포즈 지오데식 로스
    l_pose, t_err, r_err = pose_geodesic_loss(poses_h, gt_pose)
    
    # 2. 가중치 및 재투영 에러 로스 (재투영 에러는 outputs에서 계산되어야 함)
    # 만약 outputs에 없으면 여기서 직접 계산 (project_kpts 활용)
    # l_weight = weight_reg_loss(weights_h, errors_h)

    # 최종 로스 조합
    total_loss = l_pose # + 0.01 * l_weight 
    
    return total_loss, t_err, r_err