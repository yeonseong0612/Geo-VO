import torch

def pose_geodesic_loss(pred_poses, gt_pose, gamma=0.8):

    if len(pred_poses.data.shape) == 2:
        pred_poses = pred_poses[None] #
        
    n_iters = pred_poses.data.shape[0]
    
    gt_pose = gt_pose[None] 


    diff = (pred_poses.inv() * gt_pose).log() 
    err = diff.norm(dim=-1).mean(dim=-1)   
    weights = gamma ** torch.arange(n_iters - 1, -1, -1, device=err.device)
    loss = (weights * err).sum()
    
    return loss

def weight_reg_loss(weight_history, reproj_errors, gamma=0.8):
    """
    weight_history: [Iter, Total_N, 2]
    reproj_errors: [Iter, Total_N, 2]
    """
    n_iters = weight_history.shape[0]

    weighted_err = (weight_history * reproj_errors.abs()).mean(dim=(1, 2))
    
    reg = -torch.log(weight_history + 1e-6).mean(dim=(1, 2))
    
    weights = gamma ** torch.arange(n_iters - 1, -1, -1, device=weighted_err.device)
    loss = (weights * (weighted_err + 0.01 * reg)).sum()
    
    return loss

def total_loss(outputs, gt_pose):
    poses_h, weights_h, errors_h = outputs
    
    l_pose = pose_geodesic_loss(poses_h, gt_pose, gamma=0.8)
    l_weight = weight_reg_loss(weights_h, errors_h, gamma=0.8)
    
    t_loss = l_pose + 0.1 * l_weight
    
    return t_loss, l_pose.detach(), l_weight.detach()