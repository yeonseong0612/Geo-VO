import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import multiprocessing as mp
from scipy.spatial import Delaunay

from src.model import VO
from src.loader import DataFactory

# DT 계산 및 파일 저장
def save_worker(task_data):
    try:
        kpts_np = task_data['kpts']           # (N, 2)
        node_features = task_data['node_features'] # (N, 258)
        sample_id = task_data['id']
        save_dir = task_data['save_dir']
        
        if len(kpts_np) < 3:
            edges_np = np.zeros((2, 0), dtype=np.int32)
        else:
            tri = Delaunay(kpts_np)
            edges = np.concatenate([
                tri.simplices[:, [0, 1]], 
                tri.simplices[:, [1, 2]], 
                tri.simplices[:, [2, 0]]
            ], axis=0)
            edges = np.sort(edges, axis=1)
            edges_np = np.unique(edges, axis=0).T 

        if edges_np.shape[1] > 0:
            src_pts = kpts_np[edges_np[0]]
            dst_pts = kpts_np[edges_np[1]]
            edge_attr = np.linalg.norm(src_pts - dst_pts, axis=1, keepdims=True)
        else:
            edge_attr = np.zeros((0, 1), dtype=np.float32)

        save_path = os.path.join(save_dir, f"{sample_id}.npz")
        np.savez_compressed(
            save_path,
            node_features=node_features.astype(np.float16),
            edges=edges_np.astype(np.int32),
            edge_attr=edge_attr.astype(np.float16),
            kpts=kpts_np.astype(np.float32)
        )
    except Exception as e:
        print(f"Error processing {sample_id}: {e}")

# 특징 추출
@torch.no_grad()
def export_precomputed_data_parallel(model, dataloader, save_dir, num_cpu_workers=8):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    pool = mp.Pool(processes=num_cpu_workers)
    print(f"Starting Parallel Export with {num_cpu_workers} CPU cores...")

    for i, batch in enumerate(tqdm(dataloader)):
        img_tensor = batch['image'].to(device)
        B = img_tensor.shape[0]
        
        kpts_tensor, desc_tensor = model.extractor(img_tensor)
        
        h, w = img_tensor.shape[-2:]
        size_tensor = torch.tensor([w, h], device=device).view(1, 1, 2)
        kpts_norm = kpts_tensor / size_tensor
        
        if desc_tensor.shape[1] == 256:
            desc_tensor = desc_tensor.transpose(1, 2)
            
        node_features = torch.cat([desc_tensor, kpts_norm], dim=-1)

        tasks = []
        for b in range(B):
            sample_id = batch['id'][b] if 'id' in batch else f"{i*B + b:06d}"
            tasks.append({
                'kpts': kpts_tensor[b].detach().cpu().numpy(),
                'node_features': node_features[b].detach().cpu().numpy(),
                'id': sample_id,
                'save_dir': save_dir
            })
        
        pool.map_async(save_worker, tasks)

    pool.close()
    pool.join()

if __name__ == "__main__":
    DATA_PATH = "/home/yskim/geovo_data/KITTI"    
    SAVE_PATH = "/home/yskim/geovo_precomputed"    
    NUM_CORES = 8                                   

    model = VO(baseline=0.54)
    
    dataset = DataFactory(DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=8, num_workers=4, shuffle=False)

    export_precomputed_data_parallel(model, dataloader, SAVE_PATH, num_cpu_workers=NUM_CORES)
    print(f"All data saved to {SAVE_PATH}")