import torch.utils.data as data
import os
import cv2
import numpy as np
import torch
from lietorch import SE3
from utils.calib import *
from scipy.spatial.transform import Rotation as R
from CFG.vo_cfg import vo_cfg as cfg

def matrix_to_7vec(matrix):
    """[4, 4] 행렬을 [x, y, z, qx, qy, qz, qw] 7차원 벡터로 변환"""
    # matrix가 텐서라면 t도 텐서입니다.
    t = matrix[:3, 3] 
    
    # 회전 행렬 부분 추출
    rot_matrix = matrix[:3, :3]
    if torch.is_tensor(rot_matrix):
        rot_matrix = rot_matrix.detach().cpu().numpy()
    
    # scipy는 numpy 배열을 입력으로 받습니다.
    quat = R.from_matrix(rot_matrix).as_quat() # [qx, qy, qz, qw] 순서 보장
    
    # t는 이미 텐서이므로 torch.from_numpy를 쓰지 않습니다.
    return torch.cat([t.float(), torch.from_numpy(quat).float().to(t.device)])

class DataFactory(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode

        # 시퀀스 리스트 결정
        seq_list = [s.strip() for s in (cfg.trainsequencelist if mode == 'train' else cfg.valsequencelist)]
        txt_file = cfg.traintxt if mode == 'train' else cfg.valtxt

        # 데이터 리스트 로드
        txt_path = os.path.join(cfg.proj_home, 'gendata', txt_file)
        self.datalist = []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] in seq_list:
                    self.datalist.append((parts[0], int(parts[1])))

        self.posesdict = {}
        self.calib = {}
        
        # 포즈 및 캘리브레이션 사전 로드
        for seq in seq_list:
            # Pose 로드
            pose_path = os.path.join(cfg.odometry_home, cfg.poses_subdir, f"{seq}.txt")
            if os.path.exists(pose_path):
                with open(pose_path) as p:
                    raw_poses = [np.fromstring(line, sep=' ').reshape(3, 4) for line in p.readlines()]
                    self.posesdict[seq] = [np.vstack([r, [0, 0, 0, 1]]) for r in raw_poses]

            # Calib 로드 (P2 사용)
            calib_path = os.path.join(cfg.odometry_home, cfg.calib_subdir, seq, 'calib.txt')
            if os.path.exists(calib_path):
                calibdata = read_calib_file(calib_path)
                P2 = np.reshape(calibdata['P2'], (3, 4))
                fx, fy, cx, cy = P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]

                img_dir = os.path.join(cfg.odometry_home, cfg.color_subdir, seq, 'image_2')
                first_img_name = sorted(os.listdir(img_dir))[0]
                first_img_path = os.path.join(img_dir, first_img_name)

                tmp_img = cv2.imread(first_img_path)
                H_raw = tmp_img.shape[0]

                cy -= (H_raw % 32)
                self.calib[seq] = np.array([fx, fy, cx, cy], dtype=np.float32)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        seq, imgnum = self.datalist[idx]

        # 1. Ground Truth Pose 계산
        pose1 = torch.from_numpy(self.posesdict[seq][imgnum]).float()
        pose2 = torch.from_numpy(self.posesdict[seq][imgnum + 1]).float()
        
        se3_1 = SE3.InitFromVec(matrix_to_7vec(pose1).unsqueeze(0))
        se3_2 = SE3.InitFromVec(matrix_to_7vec(pose2).unsqueeze(0))
        rel_se3 = se3_1.inv() * se3_2
        rel_pose_7vec = rel_se3.data.squeeze()[:7]
        
        data = {
            'rel_pose': rel_pose_7vec,
            'calib': torch.from_numpy(self.calib[seq]).float(),
            'seq': seq,
            'imgnum': imgnum
        }
        
        # 2. 전처리된 Pair 데이터 로드 (Train 모드)
        if self.mode == 'train':
            pair_file = f"pair_{str(imgnum).zfill(6)}_{str(imgnum+1).zfill(6)}.npz"
            pair_path = os.path.join(self.cfg.precomputed_dir, seq, 'image_2', pair_file)
            
            pair_data = np.load(pair_path)
            
            data.update({
                'kpts': torch.from_numpy(pair_data['kpts']).float(),           # [800, 2]
                'pts_3d': torch.from_numpy(pair_data['pts_3d']).float(),       # [800, 3]
                'descs': torch.from_numpy(pair_data['descs']).float(),         # [800, 256]
                'temporal_matches': torch.from_numpy(pair_data['temporal_matches']).long(), # [M, 2]
                'match_scores': torch.from_numpy(pair_data['match_scores']).float(),       # [M]
                'tri_indices': torch.from_numpy(pair_data['tri_indices']).long(),         # [T, 3]
                'mask': torch.from_numpy(pair_data['mask']).bool()             # [800]
            })
            
        else:
            img_paths = [
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_2', f"{str(imgnum).zfill(6)}.png"),
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_3', f"{str(imgnum).zfill(6)}.png"),
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_2', f"{str(imgnum+1).zfill(6)}.png"),
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_3', f"{str(imgnum+1).zfill(6)}.png")
            ]

            imgs = []
            for path in img_paths:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H, W, _ = img.shape
                img = img[H % 32:, :1216]
                if img.shape[0] != 352 or img.shape[1] != 1216:
                    img = cv2.resize(img, (1216, 352), interpolation=cv2.INTER_LINEAR)
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                imgs.append(img)
                
            data['imgs'] = torch.stack(imgs) # [4, 3, 352, 1216]
            
        return data
    

def vo_collate_fn(batch):
    result = {
        'calib': torch.stack([item['calib'] for item in batch]),
        'seq': [item['seq'] for item in batch],
        'imgnum': [item['imgnum'] for item in batch]
    }
    
    if 'rel_pose' in batch[0]:
        result['rel_pose'] = torch.stack([item['rel_pose'] for item in batch])

    # --- Train 모드 전용 데이터 처리 ---
    if 'kpts' in batch[0]:
        # 크기가 고정된 [800, ...] 데이터는 stack
        result['kpts'] = torch.stack([item['kpts'] for item in batch])
        result['pts_3d'] = torch.stack([item['pts_3d'] for item in batch])
        result['descs'] = torch.stack([item['descs'] for item in batch])
        result['mask'] = torch.stack([item['mask'] for item in batch])
        
        # [중요] 매칭 쌍(M)과 삼각형(T)은 배치마다 수가 다르므로 리스트로 유지
        result['temporal_matches'] = [item['temporal_matches'] for item in batch]
        result['match_scores'] = [item['match_scores'] for item in batch]
        result['tri_indices'] = [item['tri_indices'] for item in batch]
    
    # --- Val/Test 모드 전용 데이터 처리 ---
    if 'imgs' in batch[0]:
        result['imgs'] = torch.stack([item['imgs'] for item in batch])

    return result