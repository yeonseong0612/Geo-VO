import torch
import numpy as np
import os
from tqdm import tqdm
from lietorch import SE3
import matplotlib.pyplot as plt
from src.model import VO
from src.loader import DataFactory
from CFG.vo_cfg import vo_cfg

import cv2

def save_matching_viz(img, last_weight, seq_name, i, save_dir):
    # 1. 원본 이미지 처리 (B, C, H, W) -> (H, W, C)
    if img.dim() == 4:
        img = img[0]
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # [0, 1] range일 경우 255 스케일링 및 uint8 변환
    if img_np.max() <= 1.0:
        img_np = (img_np * 255).astype(np.uint8)
    
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    H, W = img_np.shape[:2]

    weight_np = last_weight[0, 0].cpu().numpy() 
    
    weight_norm = cv2.normalize(weight_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(weight_norm, cv2.COLORMAP_JET)

    heatmap = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_LINEAR)

    combined = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    # 5. 저장
    file_path = f"{save_dir}/{seq_name}_{i:06d}.png"
    cv2.imwrite(file_path, combined)


def compute_ate_rpe(pred_poses, gt_poses):
    ate_trans = []
    rpe_trans = []
    rpe_rot = []
    
    for i in range(len(pred_poses)):
        # 1. ATE 계산: 절대적인 궤적 차이
        error_pose = gt_poses[i].inv() * pred_poses[i]
        # .translation() 메서드를 써서 [T, Q] 순서 문제를 안전하게 회피
        ate_trans.append(error_pose.translation().norm().item())
        
        if i > 0:
            # 2. RPE 계산: 프레임 간 상대적 차이
            rel_gt = gt_poses[i-1].inv() * gt_poses[i]
            rel_pred = pred_poses[i-1].inv() * pred_poses[i]
            rel_error = rel_gt.inv() * rel_pred
            
            # 병진 오차 (m)
            rpe_trans.append(rel_error.translation().norm().item())
            
            # 회전 오차 (deg): log map [vx, vy, vz, wx, wy, wz]에서 뒤의 3차원 추출
            log_err = rel_error.log() # [1, 6]
            rpe_rot.append(np.rad2deg(log_err[0, 3:].norm().item()))
            
    return np.mean(ate_trans), np.mean(rpe_trans), np.mean(rpe_rot)

def plot_trajectory(pred, gt, name, save_dir):
    # 각 포즈에서 x, y, z 좌표 추출
    pred_xyz = torch.stack([p.translation()[0] for p in pred]).cpu().numpy()
    gt_xyz = torch.stack([p.translation()[0] for p in gt]).cpu().numpy()
    
    plt.figure(figsize=(12, 9))
    # KITTI는 보통 x-z 평면 주행이므로 x와 z축을 시각화
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', label='Ground Truth', linewidth=2.0)
    plt.plot(pred_xyz[:, 0], pred_xyz[:, 2], 'r--', label='Predicted (GEO-VO)', linewidth=2.0)
    
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Z [m]', fontsize=12)
    plt.title(f"Trajectory Visualization - KITTI Sequence {name}", fontsize=15)
    plt.legend(fontsize=12)
    
    png_path = os.path.join(save_dir, f"trajectory_{name}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close() # 메모리 해제
    print(f"==> Plot saved to: {png_path}")

def evaluate(model_path, cfg, seq_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 모델 로드
    model = VO(cfg).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. 데이터셋 준비 (검증 모드)
    cfg.valsequencelist = [seq_name]
    dataset = DataFactory(cfg, mode='val')
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    save_dir = os.path.join(os.path.dirname(model_path), "eval_results", seq_name)
    os.makedirs(save_dir, exist_ok=True)

    pred_poses_list = []
    gt_poses_list = []
    kitti_poses = []
    
    # 초기 포즈 설정
    curr_pred = SE3.Identity(1, device=device)
    curr_gt = SE3.Identity(1, device=device)
    
    pred_poses_list.append(curr_pred)
    gt_poses_list.append(curr_gt)
    kitti_poses.append(curr_pred.matrix()[0, :3, :4].cpu().numpy().reshape(-1))

    print(f"==> Starting Evaluation: Sequence {seq_name}")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Inference")):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # 2. 이미 batch가 GPU에 있으므로 .to(device) 없이 사용 가능
            rel_gt_se3 = SE3.InitFromVec(batch['rel_pose'])
            
            # 3. 모델 추론 (이제 모든 텐서가 GPU에 있어 안전함)
            outputs = model(batch, iters=8, mode='test')
            rel_pred_se3 = outputs['poses'][-1]

            # --- 시각화 로직 추가 ---
            if i % 10 == 0:  # 10프레임마다 저장
                # 모델이 가중치 맵을 'weights'라는 키로 반환한다고 가정 (GEO-VO/DROID-SLAM 구조)
                if 'weights' in outputs:
                    # 마지막 반복(iteration)의 가중치 추출
                    last_weight = outputs['weights'][-1] # [1, H, W]
                    current_img = batch['imgs'][0] # [3, H, W]
                    
                    save_matching_viz(current_img, last_weight, seq_name, i, save_dir)
            # -----------------------


            # World 좌표계 누적
            curr_gt = curr_gt * rel_gt_se3
            curr_pred = curr_pred * rel_pred_se3

            gt_poses_list.append(curr_gt)
            pred_poses_list.append(curr_pred)
            
            # KITTI 형식으로 포즈 행렬 저장
            pose_mat = curr_pred.matrix()[0, :3, :4].cpu().numpy().reshape(-1)
            kitti_poses.append(pose_mat)

    # 결과 파일 저장
    txt_path = os.path.join(save_dir, f"poses_{seq_name}.txt")
    np.savetxt(txt_path, np.array(kitti_poses), fmt='%.6e')

    # 오차 분석
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

    # 궤적 그리기
    plot_trajectory(pred_poses_list, gt_poses_list, seq_name, save_dir)

if __name__ == "__main__":
    # 방금 수정한 [T, Q] 순서로 학습된 최신 체크포인트 경로를 입력하세요!
    MODEL_FILE = "checkpoint/GEO-VO/vo_model_59.pth"
    SEQUENCE = "09"
    
    if os.path.exists(MODEL_FILE):
        evaluate(MODEL_FILE, vo_cfg, SEQUENCE)
    else:
        print(f"Error: Model file not found at {MODEL_FILE}")