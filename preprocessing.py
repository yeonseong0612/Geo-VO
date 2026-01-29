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
from src.extractor import SuperPointExtractor
from src.matcher import LightGlueMatcher

def read_calib_file(path):
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            if not line.strip(): continue
            key, value = line.split(':', 1)
            data[key] = np.array([float(x) for x in value.split()])
    return data

class PreprocessDataset(Dataset):
    def __init__(self, data_root, sequences):
        self.data_root = data_root
        self.samples = []
        self.calib_dict = {}  

        for seq in sequences:
            calib_path = os.path.join(data_root, seq, 'calib.txt')
            if not os.path.exists(calib_path): continue
            
            calib_data = read_calib_file(calib_path)
            P2 = np.reshape(calib_data['P2'], (3, 4))
            fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

            img_dir = os.path.join(data_root, seq, 'image_2')
            img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
            
            first_img = cv2.imread(os.path.join(img_dir, img_names[0]))
            H_raw = first_img.shape[0]
            cy_corrected = cy - (H_raw % 32)
            
            self.calib_dict[seq] = np.array([
                [fx, 0,  cx],
                [0,  fy, cy_corrected],
                [0,  0,  1]
            ], dtype=np.float32)

            for name in img_names:
                self.samples.append({
                    'seq': seq,
                    'imgnum': int(name.split('.')[0]),
                    'img_path_2': os.path.join(data_root, seq, 'image_2', name),
                    'img_path_3': os.path.join(data_root, seq, 'image_3', name)
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs_processed = []
        for p in [s['img_path_2'], s['img_path_3']]:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape
            img = img[H % 32:, :1216] 
            imgs_processed.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        
        return {
            'images': torch.stack(imgs_processed), 
            'seq': s['seq'],
            'imgnum': s['imgnum'],
            'K': self.calib_dict[s['seq']] 
        }

class GeoVOPreprocess:
    def __init__(self, extractor, matcher, baseline=0.54):
        self.extractor = extractor
        self.matcher = matcher
        self.baseline = baseline

    def compute_3d(self, kpts_l, kpts_r, matches_dict, K):
        matches = matches_dict['matches'][0].cpu().numpy()
        idx_l, idx_r = matches[:, 0], matches[:, 1]

        # 1. Disparity 계산
        disp = kpts_l[idx_l, 0] - kpts_r[idx_r, 0]
        y_error = np.abs(kpts_l[idx_l, 1] - kpts_r[idx_r, 1])

        # 2. 강력한 필터링 (음수 및 0에 가까운 시차 제거)
        # disp < 2.0이면 너무 멀거나(Z > 150m) 오매칭일 확률이 높음
        valid = (disp > 2.0) & (y_error < 2.0)
        
        idx_l_v = idx_l[valid]
        disp_v = disp[valid]

        # [핵심] 초기값을 (0, 0, 0)이 아닌 안전한 양수 혹은 0으로 세팅
        # 나중에 모델에서 1/Z 연산을 할 때 0이나 음수가 들어오면 폭발하므로 
        # 사용하지 않는 점들은 기본적으로 0으로 둡니다.
        pts_3d = np.zeros((kpts_l.shape[0], 3), dtype=np.float32)
        
        fx, cx, cy = K[0, 0], K[0, 2], K[1, 2]
        z = (fx * self.baseline) / disp_v
        
        # 3. 물리적으로 가능한 거리(0.5m ~ 100m)만 통과
        z_mask = (z > 0.5) & (z < 100.0)
        final_idx = idx_l_v[z_mask]
        final_z = z[z_mask]

        pts_3d[final_idx, 0] = (kpts_l[final_idx, 0] - cx) * final_z / fx
        pts_3d[final_idx, 1] = (kpts_l[final_idx, 1] - cy) * final_z / fx
        pts_3d[final_idx, 2] = final_z

        return pts_3d, set(final_idx.tolist())

    @torch.no_grad()
    def process_pair(self, batch_t, batch_tp1):
        # ... (추출 및 매칭 로직은 동일) ...
        k_t_l, d_t_l = self.extractor(batch_t['images'][:, 0])
        k_t_r, d_t_r = self.extractor(batch_t['images'][:, 1])
        k_tp1_l, d_tp1_l = self.extractor(batch_tp1['images'][:, 0])

        stereo_matches = self.matcher(
            {'keypoints': k_t_l, 'descriptors': d_t_l},
            {'keypoints': k_t_r, 'descriptors': d_t_r}
        )
        temporal_matches_dict = self.matcher(
            {'keypoints': k_t_l, 'descriptors': d_t_l},
            {'keypoints': k_tp1_l, 'descriptors': d_tp1_l}
        )

        K = batch_t['K'][0].cpu().numpy()
        kpts_t_np = k_t_l[0].cpu().numpy()
        kpts_tp1_raw = k_tp1_l[0].cpu().numpy()
        temp_matches = temporal_matches_dict['matches'][0].cpu().numpy()
        
        # [수정] 3D 점 계산 및 유효 인덱스 확보
        pts_3d, valid_3d_indices = self.compute_3d(kpts_t_np, k_t_r[0].cpu().numpy(), stereo_matches, K)

        # [수정] Temporal 대응점 정렬 및 유효성 체크
        kpts_tp1_aligned = np.zeros_like(kpts_t_np)
        valid_temp_indices = set()
        if temp_matches.shape[0] > 0:
            idx_t = temp_matches[:, 0]
            idx_tp1 = temp_matches[:, 1]
            kpts_tp1_aligned[idx_t] = kpts_tp1_raw[idx_tp1]
            valid_temp_indices = set(idx_t.tolist())

        # [핵심] 최종 마스크: (3D 점이 있고) AND (다음 프레임 대응점도 있는 점)
        final_valid_mask = np.array([
            (i in valid_3d_indices) and (i in valid_temp_indices) 
            for i in range(kpts_t_np.shape[0])
        ], dtype=bool)

        # [수정] 삼각형 인덱스는 "진짜 유효한 점"들로만 구성
        tri_indices = np.array([], dtype=np.int32)
        if np.sum(final_valid_mask) >= 3:
            # 유효한 점들의 인덱스 리스트
            valid_idx_list = np.where(final_valid_mask)[0]
            # Delaunay는 유효한 점들의 좌표만 가지고 수행
            dt = Delaunay(kpts_t_np[final_valid_mask])
            # 결과 인덱스를 다시 원본 800개 기준 인덱스로 변환
            tri_indices = valid_idx_list[dt.simplices].astype(np.int32)

        return {
            'kpts': kpts_t_np,
            'pts_3d': pts_3d,
            'descs': d_t_l[0].cpu().numpy(),
            'kpts_tp1': kpts_tp1_aligned,
            'temporal_matches': temp_matches,
            'match_scores': temporal_matches_dict['scores'][0].cpu().numpy(),
            'mask': final_valid_mask, # <--- 이제 진짜 마스크 역할을 합니다.
            'tri_indices': tri_indices,
            'K': K
        }

def save_worker_pair(task_data):
    try:
        res = task_data['result']
        full_save_path = os.path.join(task_data['save_dir'], task_data['rel_path'] + ".npz")
        os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
        np.savez_compressed(
            full_save_path,
            kpts=res['kpts'].astype(np.float32),
            pts_3d=res['pts_3d'].astype(np.float32),
            descs=res['descs'].astype(np.float16),
            kpts_tp1=res['kpts_tp1'].astype(np.float32),
            temporal_matches=res['temporal_matches'].astype(np.int32),
            match_scores=res['match_scores'].astype(np.float32),
            mask=res['mask'].astype(np.bool_),
            tri_indices=res['tri_indices'].astype(np.int32),
            K=res['K'].astype(np.float32)
        )
    except Exception as e:
        print(f"저장 에러: {e}")

@torch.no_grad()
def export_parallel(extractor, matcher, dataloader, save_dir, num_cpu):
    processor = GeoVOPreprocess(extractor, matcher)
    pool = mp.Pool(processes=num_cpu)
    
    it = iter(dataloader)
    
    try:
        batch_t = next(it)
    except StopIteration:
        return

    for i in tqdm(range(len(dataloader) - 1), desc="Pair 전처리 중"):
        try:
            batch_tp1 = next(it)
        except StopIteration:
            break
            
        if batch_t['seq'][0] != batch_tp1['seq'][0]:
            batch_t = batch_tp1
            continue
            
        result = processor.process_pair(batch_t, batch_tp1)
        
        rel_path = os.path.join(batch_t['seq'][0], f"pair_{int(batch_t['imgnum'][0]):06d}_{int(batch_tp1['imgnum'][0]):06d}")
        pool.apply_async(save_worker_pair, ({'result': result, 'rel_path': rel_path, 'save_dir': save_dir},))

        batch_t = batch_tp1

    pool.close()
    pool.join()

if __name__ == "__main__":
    RAW_DATA_PATH = "/home/jnu-ie/Dataset/kitti_odometry/data_odometry_color/dataset/sequences" 
    SAVE_PATH = "/home/jnu-ie/kys/Geo-VO/gendata/precomputed"
    SEQUENCES = [f"{i:02d}" for i in range(9)] 
    
    extractor = SuperPointExtractor(max_keypoints=1000)
    matcher = LightGlueMatcher(feature_type='superpoint')

    dataset = PreprocessDataset(RAW_DATA_PATH, SEQUENCES)
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        export_parallel(extractor, matcher, dataloader, SAVE_PATH, num_cpu=vo_cfg.num_cpu)
        print(f"✨ Geo-VO 통합 전처리 완료!")