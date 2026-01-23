import torch
import numpy as np
import os
from tqdm import tqdm
from lietorch import SE3
import matplotlib.pyplot as plt
from src.model import VO
from src.loader import DataFactory
from CFG.vo_cfg import vo_cfg

def compute_ate_rpe(pred_poses, gt_poses):
    ate_trans = []
    rpe_trans = []
    rpe_rot = []
    for i in range(len(pred_poses)):
        # SE(3) 군 연산을 통한 ATE 계산
        error_pose = gt_poses[i].inv() * pred_poses[i]
        ate_trans.append(error_pose.data[0, :3].norm().item())
        if i > 0:
            rel_gt = gt_poses[i-1].inv() * gt_poses[i]
            rel_pred = pred_poses[i-1].inv() * pred_poses[i]
            rel_error = rel_gt.inv() * rel_pred
            rpe_trans.append(rel_error.data[0, :3].norm().item())
            # Lie Algebra (log map)을 이용한 회전 오차 추출
            log_err = rel_error.log()
            rpe_rot.append(np.rad2deg(log_err[0, 3:].norm().item()))
    return np.mean(ate_trans), np.mean(rpe_trans), np.mean(rpe_rot)

def evaluate(model_path, cfg, seq_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VO(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    cfg.valsequencelist = [seq_name]
    dataset = DataFactory(cfg, mode='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    pred_poses_list = []
    gt_poses_list = []
    # KITTI 포맷(SE3 Matrix) 저장을 위한 리스트
    kitti_poses = []
    
    curr_pred = SE3.Identity(1, device=device)
    curr_gt = SE3.Identity(1, device=device)
    
    # 초기 포즈 저장
    pred_poses_list.append(curr_pred)
    gt_poses_list.append(curr_gt)
    kitti_poses.append(curr_pred.matrix()[0, :3, :4].cpu().numpy().reshape(-1))

    print(f"==> Evaluating Sequence: {seq_name}")
    with torch.no_grad():
        for batch in tqdm(loader):
            rel_gt_se3 = SE3.InitFromVec(batch['rel_pose'].to(device))
            
            # 모델 예측 및 SE(3) 누적 업데이트
            poses_h, _, _ = model(batch, iters=12)
            rel_pred_se3 = SE3.InitFromVec(poses_h.data[-1])

            # G(k+1) = G(k) * Delta_T (리 군 우측 합성)
            curr_gt = curr_gt * rel_gt_se3
            curr_pred = curr_pred * rel_pred_se3

            gt_poses_list.append(curr_gt)
            pred_poses_list.append(curr_pred)
            
            # SE(3) 원소를 3x4 행렬로 변환하여 저장
            pose_mat = curr_pred.matrix()[0, :3, :4].cpu().numpy().reshape(-1)
            kitti_poses.append(pose_mat)

    # 결과 파일 저장
    txt_filename = f"pred_{seq_name}.txt"
    np.savetxt(txt_filename, np.array(kitti_poses), fmt='%.6e')
    print(f"==> Saved KITTI format trajectory to {txt_filename}")

    ate, rpe_t, rpe_r = compute_ate_rpe(pred_poses_list, gt_poses_list)
    print(f"\nResults: ATE: {ate:.4f}, RPE-t: {rpe_t:.4f}, RPE-r: {rpe_r:.4f}")
    
    plot_trajectory(pred_poses_list, gt_poses_list, seq_name)

def plot_trajectory(pred, gt, name):
    pred_xyz = torch.stack([p.data[0, :3] for p in pred]).cpu().numpy()
    gt_xyz = torch.stack([p.data[0, :3] for p in gt]).cpu().numpy()
    plt.figure(figsize=(10, 8))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', label='GT')
    plt.plot(pred_xyz[:, 0], pred_xyz[:, 2], 'r--', label='Pred')
    plt.axis('equal'); plt.legend(); plt.title(f"Seq {name}"); plt.show()

if __name__ == "__main__":
    evaluate("checkpoint/GEO-VO/vo_model_59.pth", vo_cfg, "09")