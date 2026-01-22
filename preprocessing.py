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
        '''
        일단 3차원 기하학 정보만으로 테스트 해보고 
        추후 코사인 유사도까지도 고려해보기
        '''
        if edges_np.shape[1] > 0:
            # 1. 소스(src)와 타겟(dst) 점의 좌표 가져오기
            src_pts = kpts_np[edges_np[0]]  # [E, 2]
            dst_pts = kpts_np[edges_np[1]]  # [E, 2]
            
            # 2. 상대 변위 계산 (dx, dy)
            diff = src_pts - dst_pts  # [E, 2]
            
            # 3. 거리 계산 (L2 Norm)
            dist = np.linalg.norm(diff, axis=1, keepdims=True)  # [E, 1]
            
            # 4. 결합: [E, 3] 차원의 속성 생성
            # [거리, dx, dy] 순서로 결합
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
        pass 

@torch.no_grad()
def export_parallel(model, dataloader, save_dir, num_cpu):
    model.eval()
    device = torch.device('cuda')
    model.to(device) 
    
    pool = mp.Pool(processes=num_cpu)
    print(f"전처리 시작 (CPU 코어: {num_cpu}개 사용)")

    async_results = []

    for batch in tqdm(dataloader):
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
                
                # 좌표 정규화
                k_norm = k / torch.tensor([w, h], device=device).float()
                
            
                if d.shape[0] == 256 and d.shape[1] != 256:
                    d = d.transpose(0, 1)
                
                node_feat = torch.cat([d, k_norm], dim=-1) 

                tasks.append({
                    'kpts': k.cpu().numpy(),
                    'node_features': node_feat.cpu().numpy(),
                    'rel_path': os.path.join(seqs[b], side_name, str(imgnums[b].item()).zfill(6)),
                    'save_dir': save_dir
                })
            
            res = pool.map_async(save_worker, tasks)
            async_results.append(res)

        if len(async_results) > 100:
            for r in async_results[:50]:
                r.wait()
            async_results = async_results[50:]

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