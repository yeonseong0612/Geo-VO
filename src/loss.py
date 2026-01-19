import torch

def pose_geodesic_loss(pred_poses, gt_pose, gamma=0.8):
    loss = 0.0
    n_iters = len(pred_poses)

    for i, pred in enumerate(pred_poses):
        diff = (pred.inv() * gt_pose).log()

        err = diff.norm(dim=-1).mean()

        weight = gamma ** (n_iters - i - 1)
        loss += weight * err
    
    loss

def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    loss = 0.0
    n_iter = len(weight_history)

    for i in range(n_iter):
        w = weight_history[i]
        e = reproj_errors[i]

        weighted_err = (w * e.abs()).mean()
        reg = -torch.log(w + 1e-6).mean()

        weight = gamma ** (n_iter -i - 1)
        loss += weight * (weighted_err + 0.01 * reg)
    
    return loss

def total_loss(outputs, gt_pose, cfg):
    poses_h, weights_h, errors_h = outputs
    l_pose = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)

    total_loss = l_pose + 0.1 * l_weight

    return total_loss, l_pose, l_weight