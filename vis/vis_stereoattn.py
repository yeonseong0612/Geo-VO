import sys
import os
# 현재 파일의 부모의 부모 폴더(Geo-VO)를 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 그 다음 임포트를 수행 (상대 경로인 .. 을 제거하고 절대 경로로 변경)
from src.model import VO
from src.loader import DataFactory, vo_collate_fn

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CFG.vo_cfg import vo_cfg

def run_vis():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VO(vo_cfg).to(device)
    model.eval()

    loader = DataLoader(DataFactory(vo_cfg, mode='train'), batch_size=1, collate_fn=vo_collate_fn)
    batch = next(iter(loader))
    
    with torch.no_grad():
        for k in ['node_features', 'kpts', 'calib']:
            batch[k] = batch[k].to(device)
        output = model.forward(batch)

    # 시각화할 점 선택 (800개 중 하나)
    idx = 400 
    q_pt = output['kpts_L'][0, idx].cpu().numpy()
    k_pts = output['kpts_R'][0].cpu().numpy()
    weights = output['attn_weights'][0, idx].cpu().numpy()

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.title("Left Image (Query)")
    plt.scatter(q_pt[0], q_pt[1], c='red', marker='x', s=100)
    plt.axhline(y=q_pt[1], color='red', linestyle='--', alpha=0.3)
    plt.gca().invert_yaxis()

    plt.subplot(1, 2, 2)
    plt.title("Right Image (Attention Weights)")
    plt.scatter(k_pts[:, 0], k_pts[:, 1], c=weights, cmap='viridis', s=30)
    plt.axhline(y=q_pt[1], color='red', linestyle='--', alpha=0.3) # 에피폴라 라인
    plt.gca().invert_yaxis()
    plt.show()

if __name__ == "__main__":
    run_vis()