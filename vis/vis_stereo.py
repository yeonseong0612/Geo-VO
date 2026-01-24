import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import VO
from src.loader import DataFactory, vo_collate_fn
from torch.utils.data import DataLoader
from CFG.vo_cfg import vo_cfg

def run_image_vis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VO(vo_cfg).to(device)
    model.eval()

    # 1. 데이터셋 설정
    dataset = DataFactory(vo_cfg, mode='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=vo_collate_fn)
    
    # 배치를 무작위로 가져옴
    batch = next(iter(loader))
    
    # --- [수정 구간: 경로 동기화] ---
    seq = batch['seq'][0]        # 예: "00"
    img_num = batch['imgnum'][0] # 예: 546
    
    # vo_cfg에 설정된 데이터 루트를 기반으로 경로 생성
    # 만약 절대 경로가 다르면 아래 base_path를 본인의 환경에 맞게 수정하세요.
    base_path = "/home/jnu-ie/Dataset/kitti_odometry/data_odometry_color/dataset"
    left_path = os.path.join(base_path, "sequences", seq, "image_2", f"{img_num:06d}.png")
    right_path = os.path.join(base_path, "sequences", seq, "image_3", f"{img_num:06d}.png")
    
    print(f"DEBUG: 시각화 프레임 정보 -> Sequence: {seq}, Frame: {img_num:06d}")

    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다.\nL: {left_path}\nR: {right_path}")

    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    # ------------------------------

    # 2. 모델 Forward 연산
    with torch.no_grad():
        for k in ['node_features', 'kpts', 'calib']:
            batch[k] = batch[k].to(device)
        output = model.forward(batch)

    # 4. 시각화 실행
    h, w, _ = left_img.shape
    combined_img = np.hstack((left_img, right_img))
    
    plt.figure(figsize=(20, 8))
    plt.imshow(combined_img)
    
    num_samples = 10
    N = output['kpts_L'].shape[1] 
    random_indices = np.random.choice(range(N), num_samples, replace=False)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, num_samples))

    for i, node_idx in enumerate(random_indices):
        # batch에서 뽑힌 정확한 그 프레임의 좌표들
        q_pt = output['kpts_L'][0, node_idx].cpu().numpy()
        weights = output['attn_weights'][0, node_idx].cpu().numpy()
        k_pts_R = output['kpts_R'][0].cpu().numpy()

        plt.scatter(q_pt[0], q_pt[1], color=colors[i], marker='x', s=40, linewidths=1)
        
        best_match_idx = np.argmax(weights)
        if weights[best_match_idx] > 0.05:
            target_pt = k_pts_R[best_match_idx]
            plt.plot([q_pt[0], target_pt[0] + w], [q_pt[1], target_pt[1]], 
                     color=colors[i], alpha=0.6, linewidth=0.8)
            plt.scatter(target_pt[0] + w, target_pt[1], color=colors[i], s=10, alpha=0.8)

    plt.title(f"Stereo Matching (Synced): Sequence {seq}, Frame {img_num:06d}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_image_vis()