import torch
import numpy as np
import os
from tqdm import tqdm
from lietorch import SE3
import matplotlib.pyplot as plt
from src.model import VO
from src.loader import DataFactory
from CFG.vo_cfg import vo_cfg

# 서버 환경(GUI 없음)에서 실행 시 아래 주석을 해제하세요.
# import matplotlib
# matplotlib.use('Agg') 

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
            # Lie Algebra (log map)을 이용한 회전 오차 추출 (라디안 -> 도 변환)
            log_err = rel_error.log()
            rpe_rot.append(np.rad2deg(log_err[0, 3:].norm().item()))
    return np.mean(ate_trans), np.mean(rpe_trans), np.mean(rpe_rot)

def plot_trajectory(pred, gt, name, save_dir):
    # 포즈 리스트에서 x, y, z 좌표 추출 (KIT티 기준 x, z가 평면)
    pred_xyz = torch.stack([p.data[0, :3] for p in pred]).cpu().numpy()
    gt_xyz = torch.stack([p.data[0, :3] for p in gt]).cpu().numpy()
    
    plt.figure(figsize=(12, 9))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', label='Ground Truth', linewidth=2.0)
    plt.plot(pred_xyz[:, 0], pred_xyz[:, 2], 'r--', label='Predicted (GEO-VO)', linewidth=2.0)
    
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Z [m]', fontsize=12)
    plt.title(f"Trajectory Visualization - KITTI Sequence {name}", fontsize=15)
    plt.legend(fontsize=12)
    
    # PNG 저장
    png_path = os.path.join(save_dir, f"trajectory_{name}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"==> Plot saved to: {png_path}")
    plt.show()

def evaluate(model_path, cfg, seq_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 로드
    model = VO(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 데이터셋 준비
    cfg.valsequencelist = [seq_name]
    dataset = DataFactory(cfg, mode='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # 3. 결과 저장 폴더 생성
    # checkpoint/GEO-VO/eval_results/09/ 형태
    save_dir = os.path.join(os.path.dirname(model_path), "eval_results", seq_name)
    os.makedirs(save_dir, exist_ok=True)

    pred_poses_list = []
    gt_poses_list = []
    kitti_poses = []
    
    # 초기 포즈 (Identity)
    curr_pred = SE3.Identity(1, device=device)
    curr_gt = SE3.Identity(1, device=device)
    
    pred_poses_list.append(curr_pred)
    gt_poses_list.append(curr_gt)
    # KITTI 포맷: 3x4 행렬을 한 줄(12개 원소)로 저장
    kitti_poses.append(curr_pred.matrix()[0, :3, :4].cpu().numpy().reshape(-1))

    print(f"==> Starting Evaluation: Sequence {seq_name}")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            # 정답 상대 포즈
            rel_gt_se3 = SE3.InitFromVec(batch['rel_pose'].to(device))
            
            # 모델 예측 (마지막 이터레이션 결과 사용)
            outputs = model(batch, iters=12)
            poses_h = outputs[0] # poses_history
            rel_pred_se3 = SE3.InitFromVec(poses_h.data[-1])

            # 포즈 누적 (World 좌표계 업데이트)
            curr_gt = curr_gt * rel_gt_se3
            curr_pred = curr_pred * rel_pred_se3

            gt_poses_list.append(curr_gt)
            pred_poses_list.append(curr_pred)
            
            # 결과물 저장용 행렬 변환
            pose_mat = curr_pred.matrix()[0, :3, :4].cpu().numpy().reshape(-1)
            kitti_poses.append(pose_mat)

    # 4. 결과 저장: KITTI txt 파일
    txt_path = os.path.join(save_dir, f"poses_{seq_name}.txt")
    np.savetxt(txt_path, np.array(kitti_poses), fmt='%.6e')
    print(f"==> KITTI trajectory saved to: {txt_path}")

    # 5. 오차 계산 및 결과 요약 저장
    ate, rpe_t, rpe_r = compute_ate_rpe(pred_poses_list, gt_poses_list)
    
    summary = (f"Evaluation Summary - Sequence {seq_name}\n"
               f"Model Path: {model_path}\n"
               f"{'='*40}\n"
               f"ATE (Absolute Trajectory Error): {ate:.4f} m\n"
               f"RPE-t (Relative Trans. Error):  {rpe_t:.4f} m\n"
               f"RPE-r (Relative Rot. Error):    {rpe_r:.4f} deg\n")
    
    print(f"\n{summary}")
    
    metrics_path = os.path.join(save_dir, f"metrics_{seq_name}.txt")
    with open(metrics_path, "w") as f:
        f.write(summary)
    print(f"==> Metrics saved to: {metrics_path}")

    # 6. 시각화 및 PNG 저장
    plot_trajectory(pred_poses_list, gt_poses_list, seq_name, save_dir)

if __name__ == "__main__":
    # 실행 경로 설정
    MODEL_FILE = "checkpoint/GEO-VO/vo_model_0.pth"
    SEQUENCE = "09"
    
    if os.path.exists(MODEL_FILE):
        evaluate(MODEL_FILE, vo_cfg, SEQUENCE)
    else:
        print(f"Error: Model file not found at {MODEL_FILE}")