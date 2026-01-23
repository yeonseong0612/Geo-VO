import torch.utils.data as data
import os
import cv2
import numpy as np
import torch
from lietorch import SE3
from utils.calib import *
from scipy.spatial.transform import Rotation as R

def matrix_to_7vec(matrix):
    # matrix: [4, 4] tensor
    t = matrix[:3, 3]
    # numpy 변환 전 .cpu() 명시 (안전성)
    rot_matrix = matrix[:3, :3].cpu().numpy()
    quat = R.from_matrix(rot_matrix).as_quat() # [x, y, z, w] 순서
    
    return torch.cat([t, torch.from_numpy(quat).float()])

class DataFactory(data.Dataset):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode

        if mode == 'train':
            txt_file = cfg.traintxt
            seq_list = [s.strip() for s in cfg.trainsequencelist]
        elif mode == 'val':
            txt_file = cfg.valtxt
            seq_list = [s.strip() for s in cfg.valsequencelist]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        txt_path = os.path.join(cfg.proj_home, 'gendata', txt_file)
        with open(txt_path) as f:
            self.datalist = []
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] in seq_list:
                    self.datalist.append((parts[0], int(parts[1])))

        self.posesdict = {}
        self.calib = {}
        
        for seq in seq_list:
            pose_path = os.path.join(cfg.odometry_home, cfg.poses_subdir, f"{seq}.txt")
            if os.path.exists(pose_path):
                with open(pose_path) as p:
                    raw_poses = [np.fromstring(line, sep=' ').reshape(3, 4) for line in p.readlines()]
                    self.posesdict[seq] = [np.vstack([r, [0, 0, 0, 1]]) for r in raw_poses]

            calib_path = os.path.join(cfg.odometry_home, cfg.calib_subdir, seq, 'calib.txt')
            if os.path.exists(calib_path):
                calibdata = read_calib_file(calib_path)
                P2 = np.reshape(calibdata['P2'], (3, 4))
                self.calib[seq] = np.array([P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]], dtype=np.float32) # [fx, fy, cx, cy]

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        seq, imgnum = self.datalist[idx]
        device = "cpu"

        # --- Relative pose calculation ---
        pose1 = torch.from_numpy(self.posesdict[seq][imgnum]).float()       # [4, 4]
        pose2 = torch.from_numpy(self.posesdict[seq][imgnum + 1]).float()   # [4, 4]

        # --- 상대 포즈 계산 (여기까지는 기존과 동일) ---
        se3_1 = SE3.InitFromVec(matrix_to_7vec(pose1).unsqueeze(0))
        se3_2 = SE3.InitFromVec(matrix_to_7vec(pose2).unsqueeze(0))
        rel_se3 = se3_1.inv() * se3_2
        

        # --- [핵심 수정] 데이터 추출 방식을 메서드 호출로 변경 ---
        # .data를 직접 쓰면 순서가 꼬일 위험이 큽니다.
        t_final = rel_se3.translation().squeeze(0)[:3]  # 무조건 앞의 3개만 (x, y, z)
        q_final = rel_se3.data.squeeze(0)[3:7]          # 인덱스를 명시적으로 지정 (qx, qy, qz, qw)

        data = {
            'rel_pose' : torch.cat([t_final, q_final], dim=0),
            'calib' : torch.from_numpy(self.calib[seq]),
            'seq' : seq,
            'imgnum' : imgnum
        }
        
        # --- Data Loader ---
        if self.mode == 'train':
            views = ['image_2', 'image_3', 'image_2', 'image_3'] # Lt, Rt, Lt1, Rt1
            indices = [imgnum, imgnum, imgnum + 1, imgnum + 1]

            precomputed = []
            for v, i in zip(views, indices):
                npz_path = os.path.join(self.cfg.precomputed_dir, seq, v, f"{str(i).zfill(6)}.npz")
                npz = np.load(npz_path)

                precomputed.append({
                    'node_features': torch.from_numpy(npz['node_features']).float(),    # [N, 258]
                    'edges': torch.from_numpy(npz['edges']).long(),                     # [2, E]
                    'edge_attr': torch.from_numpy(npz['edge_attr']).float(),            # [E, 1]
                    'kpts': torch.from_numpy(npz['kpts']).float()                       # [N, 2]
                })
            data['precomputed'] = precomputed
        else:
            # 2-B. 원본 PNG 로드 (기존 사용자님 코드 활용)
            paths = [
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_2', f"{str(imgnum).zfill(6)}.png"),
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_3', f"{str(imgnum).zfill(6)}.png"),
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_2', f"{str(imgnum + 1).zfill(6)}.png"),
                os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq, 'image_3', f"{str(imgnum + 1).zfill(6)}.png")
            ]

            imgs = []
            for p in paths:
                img = cv2.imread(p)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                H, W, _ = img.shape
                img = img[H % 32:, :1216]
                imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            
            data['images'] = torch.stack(imgs) # [4, 3, H, W]
        
        return data
                

def vo_collate_fn(batch):
    rel_poses = torch.stack([item['rel_pose'] for item in batch])
    calibs = torch.stack([item['calib'] for item in batch])

    all_node_features = []
    all_edges = []
    all_edge_attrs = []
    all_kpts = []
    
    # 엣지 개수가 달라도 합칠 수 있도록 cat 방식을 사용하되, 
    # 나중에 뷰별로 분리할 수 있게 구조를 유지합니다.
    for item in batch:
        for s in item['precomputed']:
            all_node_features.append(s['node_features']) # [800, 256]
            all_edges.append(s['edges'])                 # [2, E_i] -> 엣지 개수 다름
            all_edge_attrs.append(s['edge_attr'])        # [E_i, 3]
            all_kpts.append(s['kpts'])                   # [800, 2]

    return {
        'rel_pose': rel_poses,
        'calib': calibs,
        'node_features': torch.stack(all_node_features), # [B*4, 800, 256] (고정 크기이므로 stack 가능)
        'edges': all_edges,                              # 리스트 형태로 유지 (개수가 다르므로)
        'edge_attr': all_edge_attrs,                     # 리스트 형태로 유지
        'kpts': torch.stack(all_kpts),                   # [B*4, 800, 2]
        'seq': [item['seq'] for item in batch],
        'imgnum': [item['imgnum'] for item in batch]
    }