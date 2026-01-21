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

        se3_1 =  SE3.InitFromVec(matrix_to_7vec(pose1).unsqueeze(0))        # [4, 4] -> [1, 7]
        se3_2 =  SE3.InitFromVec(matrix_to_7vec(pose2).unsqueeze(0))        # [4, 4] -> [1, 7]
        rel_pose = (se3_1.inv() * se3_2).data.squeeze(0) # [tx, ty, tz, qx, qy, qz, qw] [1, 7] -> [4, 4]

        data = {
            'rel_pose' : rel_pose,
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
                

import torch.nn.functional as F

def vo_collate_fn(batch):
    rel_poses = torch.stack([item['rel_pose'] for item in batch])
    calibs = torch.stack([item['calib'] for item in batch])
    seqs = [item['seq'] for item in batch]
    imgnums = [item['imgnum'] for item in batch]

    MAX_KPT = 800 

    if 'precomputed' in batch[0]:
        all_node_features = []
        all_edges = []
        all_edge_attrs = []
        all_kpts_fixed = [] # 고정 크기 리스트
        
        node_offset = 0
        
        for item in batch:
            for s in item['precomputed']:
                k = s['kpts'] # [N, 2]
                f = s['node_features'] # [N, 258]
                
                num_k = k.shape[0]
                if num_k > MAX_KPT:
                    k = k[:MAX_KPT]
                    f = f[:MAX_KPT]
                elif num_k < MAX_KPT:
                    k_pad = torch.zeros((MAX_KPT - num_k, 2), device=k.device)
                    f_pad = torch.zeros((MAX_KPT - num_k, 258), device=f.device)
                    k = torch.cat([k, k_pad], dim=0)
                    f = torch.cat([f, f_pad], dim=0)

                all_kpts_fixed.append(k)
                all_node_features.append(f)

                all_edges.append(s['edges'] + node_offset)
                all_edge_attrs.append(s['edge_attr'])
                node_offset += MAX_KPT 

        return {
            'rel_pose': rel_poses,
            'calib': calibs,
            'node_features': torch.stack(all_node_features), # [B*4, MAX_KPT, 258]
            'edges': torch.cat(all_edges, dim=1),
            'edge_attr': torch.cat(all_edge_attrs, dim=0),
            'kpts': torch.stack(all_kpts_fixed).view(len(batch), 4, MAX_KPT, 2),
            'seq': seqs,
            'imgnum': imgnums
        }
    
    else:
        images = torch.stack([item['images'] for item in batch]) 
        return {
            'images': images,
            'rel_pose': rel_poses,
            'calib': calibs,
            'seq': seqs,
            'imgnum': imgnums
        }