import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing as mp

from src.model import VO
from src.loader import DataFactory
from CFG.vo_cfg import vo_cfg  

def save_worker(task_data):
    try:
        from scipy.spatial import Delaunay
        kpts_np = task_data['kpts']
        node_features = task_data['node_features']
        rel_path = task_data['rel_path'] 
        save_dir = task_data['save_dir']
        
        full_save_path = os.path.join(save_dir, rel_path + ".npz")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

        if len(kpts_np) < 3:
            edges_np = np.zeros((2, 0), dtype=np.int32)
        else:
            tri = Delaunay(kpts_np)
            edges = np.concatenate([tri.simplices[:, [0, 1]], tri.simplices[:, [1, 2]], tri.simplices[:, [2, 0]]], axis=0)
            edges = np.sort(edges, axis=1)
            edges_np = np.unique(edges, axis=0).T

        if edges_np.shape[1] > 0:
            edge_attr = np.linalg.norm(kpts_np[edges_np[0]] - kpts_np[edges_np[1]], axis=1, keepdims=True)
        else:
            edge_attr = np.zeros((0, 1), dtype=np.float32)

        np.savez_compressed(
            full_save_path,
            node_features=node_features.astype(np.float16),
            edges=edges_np.astype(np.int32),
            edge_attr=edge_attr.astype(np.float16),
            kpts=kpts_np.astype(np.float32)
        )
    except Exception as e:
        pass 

@torch.no_grad()
def export_parallel(model, dataloader, save_dir, num_cpu):
    model.eval()
    device = torch.device('cuda')
    model.to(device) 
    
    # 1. Pool을 루프 밖에서 한 번만 관리
    pool = mp.Pool(processes=num_cpu)
    print(f"전처리 시작 (CPU 코어: {num_cpu}개 사용)")

    for batch in tqdm(dataloader):
        images = batch['images'].to(device)
        B = images.shape[0]
        seqs = batch['seq']
        imgnums = batch['imgnum']

        h, w = images.shape[-2:]

        for side_idx, side_name in zip([0, 1], ['image_2', 'image_3']):
            # GPU 연산 (추출)
            kpts_list, desc_list = model.extractor(images[:, side_idx])
            
            tasks = []
            for b in range(B):
                k = kpts_list[b]      # [N, 2]
                d = desc_list[b]      # [256, N]
                
                k_norm = k / torch.tensor([w, h], device=device).float()
                
                if d.shape[0] == 256:
                    d = d.transpose(0, 1)
                node_feat = torch.cat([d, k_norm], dim=-1)

                tasks.append({
                    'kpts': k.cpu().numpy(),
                    'node_features': node_feat.cpu().numpy(),
                    'rel_path': os.path.join(seqs[b], side_name, str(imgnums[b].item()).zfill(6)),
                    'save_dir': save_dir
                })
            
            pool.map_async(save_worker, tasks)

    pool.close()
    pool.join()

if __name__ == "__main__":
    SAVE_PATH = "/home/jnu-ie/kys/Geo-VO/geovo_prcomputed"
    
    model = VO(baseline=0.54).cuda()
    print("모델 로드 완료")

    dataset = DataFactory(vo_cfg, mode='train')
    dataloader = DataLoader(dataset, batch_size=vo_cfg.batchsize, num_workers=4, shuffle=False)
    print(f"데이터셋 발견. 총 프레임: {len(dataset)}")

    export_parallel(model, dataloader, SAVE_PATH, num_cpu=vo_cfg.num_cpu)
    print(f"전처리 완료! 저장 위치: {SAVE_PATH}")