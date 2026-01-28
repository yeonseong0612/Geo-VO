import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import multiprocessing as mp
from scipy.spatial import Delaunay

from src.model import VO
from CFG.vo_cfg import vo_cfg

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def read_calib_file(path):
    """KITTI calib.txt 파일을 읽어 딕셔너리로 반환합니다."""
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = np.array([float(x) for x in value.split()])
    return data

class PreprocessDataset(Dataset):
    def __init__(self, data_root, sequences):
        self.data_root = data_root
        self.samples = []
        self.calib_dict = {}  

        for seq in sequences:
            calib_path = os.path.join(data_root, seq, 'calib.txt')
            if not os.path.exists(calib_path):
                print(f"⚠️ {seq}: calib.txt를 찾을 수 없습니다.")
                continue
            
            calib_data = read_calib_file(calib_path)
            P2 = np.reshape(calib_data['P2'], (3, 4))
            fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

            img_dir = os.path.join(data_root, seq, 'image_2')
            if not os.path.exists(img_dir): continue
            
            img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
            first_img = cv2.imread(os.path.join(img_dir, img_names[0]))
            H_raw = first_img.shape[0]

            cy_corrected = cy - (H_raw % 32)
            
            self.calib_dict[seq] = np.array([
                [fx, 0,  cx],
                [0,  fy, cy_corrected],
                [0,  0,  1]
            ], dtype=np.float32)

            # 4. 샘플 리스트 생성
            for name in img_names:
                self.samples.append({
                    'seq': seq,
                    'imgnum': int(name.split('.')[0]),
                    'img_path_2': os.path.join(data_root, seq, 'image_2', name),
                    'img_path_3': os.path.join(data_root, seq, 'image_3', name)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq = s['seq']
        imgs_processed = []
        
        for p in [s['img_path_2'], s['img_path_3']]:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape
            
            top_crop = H % 32
            img = img[top_crop:, :1216]
            
            
            imgs_processed.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        
        return {
            'images': torch.stack(imgs_processed), 
            'seq': seq,
            'imgnum': s['imgnum'],
            'K': self.calib_dict[seq] 
        }

def save_worker(task_data):
    try:
        kpts = task_data['kpts'] # [N, 2]
        descs = task_data['node_features'] # [N, 256]
        scores = task_data['scores'] # [N]
        
        # 1. 유효성 마스크 생성 (패딩된 0좌표 제외)
        mask = np.linalg.norm(kpts, axis=1) > 0
        
        # 2. Delaunay Triangulation 및 삼각형 법선 벡터 (R 추정용)
        valid_kpts = kpts[mask]
        tri_normals = np.array([])
        tri_indices = np.array([])
        
        if len(valid_kpts) >= 3:
            dt = Delaunay(valid_kpts)
            tri_indices = dt.simplices
            # 단순 예시: 삼각형의 외적을 통한 가상 법선 벡터 계산 (소실점 투표 활용)
            # 실제 연구 수식에 따라 이 부분을 tri_normals 계산 로직으로 교체하세요.
            tri_normals = np.zeros((len(tri_indices), 3)) 

        full_save_path = os.path.join(task_data['save_dir'], task_data['rel_path'] + ".npz")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)

        np.savez_compressed(
            full_save_path,
            kpts=kpts.astype(np.float32),
            descs=descs.astype(np.float16),
            scores=scores.astype(np.float32),
            mask=mask.astype(np.bool_),
            tri_indices=tri_indices.astype(np.int32),
            tri_normals=tri_normals.astype(np.float32),
            K=task_data['K'].astype(np.float32),
            image_size=np.array([352, 1216], dtype=np.float32)
        )
    except Exception as e:
        print(f"❌ 저장 에러: {e}")

# --- [3] 병렬 추출 루프 ---
@torch.no_grad()
def export_parallel(model, dataloader, save_dir, num_cpu):
    model.eval()
    device = torch.device('cuda')
    model.to(device)
    pool = mp.Pool(processes=num_cpu)
    async_results = []

    for batch in tqdm(dataloader, desc="⚙️ 통합 전처리 중"):
        images = batch['images'].to(device)
        B = images.shape[0]
        K_batch = batch['K'].numpy()

        for side_idx, side_name in zip([0, 1], ['image_2', 'image_3']):
            # [수정] 모델에서 점수(scores)까지 함께 추출하도록 인터페이스 확인 필요
            kpts, descs = model.extractor(images[:, side_idx])
            # SuperPoint에서 score를 따로 뽑는 로직이 없다면 임의의 1.0으로 초기화 가능
            scores = torch.ones(B, kpts.shape[1], device=device) 

            tasks = []
            for b in range(B):
                rel_path = os.path.join(batch['seq'][b], side_name, f"{int(batch['imgnum'][b]):06d}")
                tasks.append({
                    'kpts': kpts[b].cpu().numpy(),
                    'node_features': descs[b].cpu().numpy(),
                    'scores': scores[b].cpu().numpy(),
                    'K': K_batch[b],
                    'rel_path': rel_path,
                    'save_dir': save_dir
                })
            
            res = pool.map_async(save_worker, tasks)
            async_results.append(res)

        if len(async_results) > 40:
            for r in async_results[:20]: r.wait()
            async_results = async_results[20:]

    pool.close()
    pool.join()

if __name__ == "__main__":
    # 경로 설정 및 실행 로직 동일
    pass