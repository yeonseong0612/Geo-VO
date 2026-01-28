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

import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp

# --- [1] 전처리 전용 데이터셋 클래스 (Crop 로직 적용) ---
class PreprocessDataset(Dataset):
    def __init__(self, data_root, sequences):
        self.data_root = data_root
        self.samples = []
        
        for seq in sequences:
            img_dir = os.path.join(data_root, seq, 'image_2')
            if not os.path.exists(img_dir): continue
            
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
        img2_raw = cv2.imread(s['img_path_2'])
        img3_raw = cv2.imread(s['img_path_3'])
        
        imgs_processed = []
        for raw_img in [img2_raw, img3_raw]:
            # 1. BGR -> RGB 변환
            img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape
            
            # 2. 원래 모델 크롭 로직: 32 배수 맞추기 및 가로 1216 제한
            # H % 32를 통해 상단을 쳐냄
            img = img[H % 32:, :1216]
            
            # 3. [중요] 시퀀스 간 미세한 세로 크기 차이 방지
            # KITTI sequences 00-08은 보통 크롭 후 352 혹은 384가 되는데, 
            # 배치를 묶기 위해 강제로 352로 맞춥니다. (대부분 352임)
            if img.shape[0] != 352 or img.shape[1] != 1216:
                img = cv2.resize(img, (1216, 352), interpolation=cv2.INTER_LINEAR)
            
            # 4. Tensor화 [H, W, C] -> [C, H, W]
            imgs_processed.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        
        # 이제 모든 이미지가 (3, 352, 1216)이므로 stack 에러가 발생하지 않음
        return {
            'images': torch.stack(imgs_processed), # [2, 3, 352, 1216]
            'seq': s['seq'],
            'imgnum': s['imgnum']
        }

# --- [1] save_worker 수정: 800개를 그대로 저장 (DT 연산 제거) ---
def save_worker(task_data):
    try:
        # 이제 전처리 단계에서 DT(Delaunay)를 하지 않습니다. 
        # Selector가 점을 고른 뒤에 학습 시 실시간으로 수행해야 하기 때문입니다.
        kpts_np = task_data['kpts']
        node_features = task_data['node_features']
        rel_path = task_data['rel_path'] 
        save_dir = task_data['save_dir']
        
        full_save_path = os.path.join(save_dir, rel_path + ".npz")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

        # 800개의 좌표와 256차원 디스크립터만 압축 저장
        np.savez_compressed(
            full_save_path,
            node_features=node_features.astype(np.float16), # 용량 절약을 위한 fp16
            kpts=kpts_np.astype(np.float32)
        )
    except Exception as e:
        print(f"Error saving {rel_path}: {e}")

@torch.no_grad()
def export_parallel(model, dataloader, save_dir, num_cpu):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device) 

    pool = mp.Pool(processes=num_cpu)
    async_results = []

    for batch in tqdm(dataloader, desc="SuperPoint Raw Extraction"):
        images = batch['images'].to(device) # [B, 2, 3, 352, 1216]
        B = images.shape[0]
        seqs = batch['seq']
        imgnums = batch['imgnum'] 

        for side_idx, side_name in zip([0, 1], ['image_2', 'image_3']):
            # [수정] 800개 원본을 그대로 뽑습니다. (Selector를 거치지 않음!)
            kpts_raw, desc_raw = model.extractor(images[:, side_idx])
            
            tasks = []
            for b in range(B):
                # 원본 800개와 256차원 유지
                k = kpts_raw[b]     
                d = desc_raw[b]   
                
                file_name = f"{int(imgnums[b]):06d}" 
                rel_path = os.path.join(seqs[b], side_name, file_name)

                tasks.append({
                    'kpts': k.cpu().numpy(),
                    'node_features': d.cpu().numpy(),
                    'rel_path': rel_path,
                    'save_dir': save_dir
                })
            
            res = pool.map_async(save_worker, tasks)
            async_results.append(res)

        # 세션 관리 (메모리 누수 방지)
        if len(async_results) > 20:
            for r in async_results[:10]: r.wait()
            async_results = async_results[10:]

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