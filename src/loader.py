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
                # [fx, fy, cx, cy]
                self.calib[seq] = np.array([P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]], dtype=np.float32)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        seq, imgnum = self.datalist[idx]

        # 1. 상대 포즈 계산 (Ground Truth)
        pose1 = torch.from_numpy(self.posesdict[seq][imgnum]).float()
        pose2 = torch.from_numpy(self.posesdict[seq][imgnum + 1]).float()
        
        se3_1 = SE3.InitFromVec(matrix_to_7vec(pose1).unsqueeze(0))
        se3_2 = SE3.InitFromVec(matrix_to_7vec(pose2).unsqueeze(0))
        rel_se3 = se3_1.inv() * se3_2
        rel_pose_7vec = rel_se3.data.squeeze()[:7]
        
        data = {
            'rel_pose': rel_pose_7vec,                          # [7]
            'calib': torch.from_numpy(self.calib[seq]),         # [4]
            'seq': seq,
            'imgnum': imgnum
        }
        
        # 2. 전처리된 데이터 로드 (Lt, Rt, Lt1, Rt1)
        # if self.mode == 'train':
            # views = ['image_2', 'image_3', 'image_2', 'image_3']
            # indices = [imgnum, imgnum, imgnum + 1, imgnum + 1]

            # precomputed = []
            # for v, i in zip(views, indices):
            #     npz_path = os.path.join(self.cfg.precomputed_dir, seq, v, f"{str(i).zfill(6)}.npz")
            #     npz = np.load(npz_path)
                
            #     # 각 프레임 데이터를 개별적으로 리스트에 담음 (collate에서 합침)
            #     precomputed.append({
            #         'node_features': torch.from_numpy(npz['node_features']).float(), # [800, 256]
            #         'edges': torch.from_numpy(npz['edges']).long(),                  # [2, E]
            #         'edge_attr': torch.from_numpy(npz['edge_attr']).float(),         # [E, 3]
            #         'kpts': torch.from_numpy(npz['kpts']).float()                    # [800, 2]
            #     })
            
            # data['precomputed'] = precomputed
        # else:
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

# def vo_collate_fn(batch):
#     # 1. 단순 텐서 스택
#     rel_poses = torch.stack([item['rel_pose'] for item in batch]) # [B, 7]
#     calibs = torch.stack([item['calib'] for item in batch])       # [B, 4]

#     # 2. 노드 및 좌표 데이터 (4차원 정렬: [B, 4, 800, C])
#     # 모델의 forward에서 [:, 0] 등으로 접근 가능하게 함
#     batch_node_features = []
#     batch_kpts = []
    
#     # 3. 에지 데이터 (가변 크기이므로 평탄화된 리스트 유지: 크기 B*4)
#     all_edges = []
#     all_edge_attrs = []
    
#     for item in batch:
#         # 각 샘플 내의 4개 뷰를 stack -> [4, 800, C]
#         sample_node_feats = torch.stack([s['node_features'] for s in item['precomputed']])
#         sample_kpts = torch.stack([s['kpts'] for s in item['precomputed']])
        
#         batch_node_features.append(sample_node_feats)
#         batch_kpts.append(sample_kpts)
        
#         # 에지는 리스트에 순차적으로 추가 (순서: 샘플0_Lt, 샘플0_Rt, 샘플0_Lt1, 샘플0_Rt1, 샘플1_Lt...)
#         for s in item['precomputed']:
#             all_edges.append(s['edges'])
#             all_edge_attrs.append(s['edge_attr'])

#     return {
#         'rel_pose': rel_poses,
#         'calib': calibs,
#         'node_features': torch.stack(batch_node_features), # [B, 4, 800, 256]
#         'kpts': torch.stack(batch_kpts),                   # [B, 4, 800, 2]
#         'edges': all_edges,                                # [B*4] 크기의 리스트
#         'edge_attr': all_edge_attrs,                       # [B*4] 크기의 리스트
#         'seq': [item['seq'] for item in batch],
#         'imgnum': [item['imgnum'] for item in batch]
#     }

def vo_collate_fn(batch):
    imgs = torch.stack([item['imgs'] for item in batch]) 
    
    calibs = torch.stack([item['calib'] for item in batch])
    
    rel_poses = torch.stack([item['rel_pose'] for item in batch]) if 'rel_pose' in batch[0] else None

    return {
        'imgs': imgs,
        'calib': calibs,
        'rel_pose': rel_poses,
        'seq': [item['seq'] for item in batch],
        'imgnum': [item['imgnum'] for item in batch]
    }