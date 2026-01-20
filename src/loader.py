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
        self.posesdict = {}
        self.intrinsicsdict = {}
        self.datalist = []

        seq_list = cfg.trainsequencelist if mode == 'train' else cfg.valsequencelist
        seq_list = [s.strip() for s in seq_list]

        for seq in seq_list:
            # Pose 로드 (KITTI 정형 데이터 포맷)
            pose_path = f"{cfg.odometry_home}{cfg.poses_subdir}{seq}.txt"
            if not os.path.exists(pose_path): continue
            
            with open(pose_path) as p:
                raw_poses = [np.fromstring(line, sep=' ').reshape(3, 4) for line in p.readlines()]
                self.posesdict[seq] = [np.vstack([r, [0, 0, 0, 1]]) for r in raw_poses]

            # Intrinsics 로드
            calib_path = f"{cfg.odometry_home}{cfg.calib_subdir}{seq}/calib.txt"
            calibdata = read_calib_file(calib_path)
            P2 = np.reshape(calibdata['P2'], (3, 4))
            # [fx, fy, cx, cy]
            self.intrinsicsdict[seq] = np.array([P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]], dtype=np.float32)

        # 데이터 리스트 생성 (프레임 페어링)
        txt_path = f"{cfg.proj_home}gendata/{cfg.traintxt if mode == 'train' else cfg.valtxt}"
        with open(txt_path) as f:
            all_lines = f.readlines()
            for line in all_lines:
                parts = line.strip().split(' ')
                if len(parts) < 2: continue
                
                seq = parts[0]
                imgnum = int(parts[1])
                
                if seq in seq_list:
                    # GT 포즈가 존재하는 범위 내에서만 추가
                    if imgnum < len(self.posesdict[seq]) - 1:
                        self.datalist.append((seq, imgnum))

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        seq, imgnum = self.datalist[idx]

        # 이미지 경로 (image_2: Left, image_3: Right)
        paths = [
            f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_2/{str(imgnum).zfill(6)}.png",
            f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_3/{str(imgnum).zfill(6)}.png",
            f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_2/{str(imgnum + 1).zfill(6)}.png",
            f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_3/{str(imgnum + 1).zfill(6)}.png"
        ]
        
        imgs = []
        for p in paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 32배수 크롭 및 가로 사이즈 고정 (Geo-VO/DROID 특성)
            H, W, _ = img.shape
            img = img[H % 32:, :1216] 
            imgs.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
        
        imgs_tensor = torch.stack(imgs) # [4, 3, H', W']

        # 상대 포즈 계산
        pose1 = torch.from_numpy(self.posesdict[seq][imgnum]).float()
        pose2 = torch.from_numpy(self.posesdict[seq][imgnum + 1]).float()

        se3_1 = SE3.InitFromVec(matrix_to_7vec(pose1).unsqueeze(0)) 
        se3_2 = SE3.InitFromVec(matrix_to_7vec(pose2).unsqueeze(0))

        # 상대 포즈: T_12 = T1^-1 * T2
        rel_pose = se3_1.inv() * se3_2

        return {
            'images': imgs_tensor,   
            'rel_pose': rel_pose.data.squeeze(0), # [7] 텐서로 변환
            'intrinsics': torch.from_numpy(self.intrinsicsdict[seq]),
            'seq': seq,
            'imgnum': imgnum
        }