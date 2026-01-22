import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
import torch.nn as nn

from src.model import VO
from CFG.vo_cfg import vo_cfg  

# --- [1] 전처리 전용 데이터셋 클래스 ---
class PreprocessDataset(Dataset):
    def __init__(self, data_root, sequences):
        self.data_root = data_root
        self.samples = []
        
        for seq in sequences:
            img_dir = os.path.join(data_root, seq, 'image_2')
            if not os.path.exists(img_dir): continue
            
            # 모든 프레임을 숫자 순서대로 정렬하여 수집
            img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
            
            for name in img_names:
                img_num = int(name.split('.')[0])
                self.samples.append({
                    'seq': seq,
                    'imgnum': img_num,
                    'img_path_2': os.path.join(data_root, seq, 'image_2', name),
                    'img_path_3': os.path.join(data_root, seq, 'image_3', name)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        # BGR -> RGB 변환 및 Tensor화
        img2 = cv2.imread(s['img_path_2'])
        img3 = cv2.imread(s['img_path_3'])
        
        img2 = torch.from_numpy(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        img3 = torch.from_numpy(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        
        return {
            'images': torch.stack([img2, img3]), # [2, 3, H, W]
            'seq': s['seq'],
            'imgnum': s['imgnum']
        }

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
            src_pts = kpts_np[edges_np[0]]
            dst_pts = kpts_np[edges_np[1]]
            diff = src_pts - dst_pts
            dist = np.linalg.norm(diff, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist, diff], axis=1)
        else:
            edge_attr = np.empty((0, 3), dtype=np.float32)

        np.savez_compressed(
            full_save_path,
            node_features=node_features.astype(np.float16),
            edges=edges_np.astype(np.int32),
            edge_attr=edge_attr.astype(np.float16),
            kpts=kpts_np.astype(np.float32)
        )
    except Exception as e:
        print(f"Error saving {rel_path}: {e}")

# ... (상단 임포트 및 PreprocessDataset 클래스 동일) ...

@torch.no_grad()
def export_parallel(model, dataloader, save_dir, num_cpu):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) 
    
    pool = mp.Pool(processes=num_cpu)
    async_results = []

    for batch in tqdm(dataloader, desc="Feature Extraction"):
        images = batch['images'].to(device) 
        B = images.shape[0]
        seqs = batch['seq']
        imgnums = batch['imgnum'] 
        h, w = images.shape[-2:]

        for side_idx, side_name in zip([0, 1], ['image_2', 'image_3']):
            kpts_all, desc_all = model.extractor(images[:, side_idx])
            
            tasks = []
            for b in range(B):
                k = kpts_all[b]     
                d = desc_all[b]   
                
                k_norm = k / torch.tensor([w, h], device=device).float()
                
                if d.shape[0] == 256 and d.shape[1] != 256:
                    d = d.transpose(0, 1)
                
                node_feat = torch.cat([d, k_norm], dim=-1) 

                # 저장 경로 설정: [시퀀스]/[이미지_사이드]/[000000].npz
                file_name = f"{int(imgnums[b]):06d}" 
                rel_path = os.path.join(seqs[b], side_name, file_name)

                tasks.append({
                    'kpts': k.cpu().numpy(),
                    'node_features': node_feat.cpu().numpy(),
                    'rel_path': rel_path,
                    'save_dir': save_dir
                })
            
            res = pool.map_async(save_worker, tasks)
            async_results.append(res)

        if len(async_results) > 50:
            for r in async_results[:25]: r.wait()
            async_results = async_results[25:]

    pool.close()
    pool.join()

if __name__ == "__main__":
    RAW_DATA_PATH = "/home/jnu-ie/Dataset/kitti_odometry/data_odometry_color/dataset/sequences" 
    SAVE_PATH = "/home/jnu-ie/kys/Geo-VO/gendata/precomputed"
    SEQUENCES = [f"{i:02d}" for i in range(9)] # 00~08
    
    vo_cfg.use_precomputed = False
    model = VO(vo_cfg).cuda()
    
    dataset = PreprocessDataset(RAW_DATA_PATH, SEQUENCES)
    print(f"🔎 찾은 데이터 샘플 수: {len(dataset)}") # 0이 나오면 경로 문제입니다.
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=vo_cfg.batchsize, num_workers=4, shuffle=False)
        export_parallel(model, dataloader, SAVE_PATH, num_cpu=vo_cfg.num_cpu)
        print(f"✨ 전처리 완료! 저장 위치: {SAVE_PATH}")
    else:
        print("❌ 데이터를 찾지 못했습니다. RAW_DATA_PATH를 다시 확인해주세요.")