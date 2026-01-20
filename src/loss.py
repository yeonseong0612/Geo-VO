import torch
import lietorch
from lietorch import SE3
def pose_geodesic_loss(pred_poses, gt_pose, gamma=0.8):
    # 1. pred_poses의 구조 파악
    # shape가 (N, 7)이면 단일 포즈 배치가 들어온 것 (Iteration 차원 없음)
    # shape가 (Iter, N, 7)이면 히스토리가 들어온 것
    data_shape = pred_poses.data.shape
    
    if len(data_shape) == 2:
        # 단일 배치 [B, 7] -> 루프를 위해 [1, B, 7]로 강제 인식
        n_iters = 1
        batch_size = data_shape[0]
        # 리스트로 감싸서 루프가 가능하게 만듦
        pred_list = [pred_poses]
    else:
        # 히스토리 [Iter, B, 7]
        n_iters = data_shape[0]
        batch_size = data_shape[1]
        pred_list = [pred_poses[i] for i in range(n_iters)]

    loss = 0.0
    
    # 2. gt_pose 배치 크기 교정
    # gt_pose.data의 크기가 28인데 batch_size가 4라면 [4, 7]로 리셰이프
    if gt_pose.data.numel() != batch_size * 7:
        # 만약 전체 데이터 개수가 맞지 않으면 에러 방지를 위해 강제 리셰이프
        actual_gt_data = gt_pose.data.view(-1, 7)[:batch_size]
    else:
        actual_gt_data = gt_pose.data.view(batch_size, 7)
    
    actual_gt = SE3(actual_gt_data)

    # 3. Geodesic Loss 계산
    for i in range(n_iters):
        pred = pred_list[i] # [B, 7]
        
        # Geodesic distance: Log(Inv(P) * G)
        diff = (pred.inv() * actual_gt).log() # [B, 6]
        err = diff.norm(dim=-1).mean()

        weight = gamma ** (n_iters - i - 1)
        loss += weight * err
    
    return loss
def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    # 일반 텐서 리스트/맵인 경우 torch.cat 사용
    if isinstance(weight_history, (map, list, tuple)):
        weight_history = torch.cat(list(weight_history), dim=1)
        reproj_errors = torch.cat(list(reproj_errors), dim=1)

    n_iter = weight_history.shape[0]
    loss = 0.0

    for i in range(n_iter):
        w = weight_history[i]
        e = reproj_errors[i]
        weighted_err = (w * e.abs()).mean()
        reg = -torch.log(w + 1e-6).mean()
        
        weight = gamma ** (n_iter - i - 1)
        loss += weight * (weighted_err + 0.01 * reg)
    
    return loss
def total_loss(outputs, gt_pose, cfg):
    # 에러 방지용 Unpacking 로직
    if len(outputs) == 2:
        # 모델이 (poses, extra) 형태로 내보낼 때
        poses_h = outputs[0]
        # 만약 extra 안에 (weights, errors)가 묶여 있다면
        if isinstance(outputs[1], (list, tuple)):
            weights_h = outputs[1][0]
            errors_h = outputs[1][1]
        else:
            # 에러가 단일 값일 경우를 대비한 방어 코드
            weights_h = outputs[1]
            errors_h = outputs[1] # 임시 할당
    else:
        # 원래 기대했던 3개 구조일 때
        poses_h, weights_h, errors_h = outputs
    
    # 이후 계산 진행
    l_pose = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)
    
    t_loss = l_pose + 0.1 * l_weight
    return t_loss, l_pose, l_weight