import torch.utils.data as data
import cv2
import numpy as np
import torch
from lietorch import SE3
from utils.calib import *
from scipy.spatial.transform import Rotation as R

def matrix_to_7vec(matrix):
    t = matrix[:3, 3]

    rot_matrix = matrix[:3, :3].numpy()
    quat = R.from_matrix(rot_matrix).as_quat() 
    
    # 3. 합치기
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

        # --- 순서 변경: 먼저 포즈와 캘리브레이션을 로드합니다 ---
        for seq in seq_list:
            # Pose 로드
            pose_path = f"{cfg.odometry_home}{cfg.poses_subdir}{seq}.txt"
            with open(pose_path) as p:
                raw_poses = [np.fromstring(line, sep=' ').reshape(3, 4) for line in p.readlines()]
                # 3x4 -> 4x4 변환
                self.posesdict[seq] = [np.vstack([r, [0, 0, 0, 1]]) for r in raw_poses]

            # Intrinsics 로드
            calib_path = f"{cfg.odometry_home}{cfg.calib_subdir}{seq}/calib.txt"
            calibdata = read_calib_file(calib_path)
            P2 = np.reshape(calibdata['P2'], (3, 4))
            self.intrinsicsdict[seq] = np.array([P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2]], dtype=np.float32)

        # --- 이제 시퀀스 길이를 알 수 있으므로 안전하게 필터링합니다 ---
        txt_path = f"{cfg.proj_home}gendata/{cfg.traintxt if mode == 'train' else cfg.valtxt}"
        with open(txt_path) as f:
            all_lines = f.readlines()
            for line in all_lines:
                parts = line.strip().split(' ')
                if not parts[0]: continue
                
                seq = parts[0]
                imgnum = int(parts[1])
                
                if seq in seq_list:
                    # 다음 프레임(imgnum + 1)이 존재하는지 확인
                    if imgnum < len(self.posesdict[seq]) - 1:
                        self.datalist.append(line)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        line = self.datalist[idx].strip().split(' ')
        seq, imgnum = line[0], int(line[1])

        path_Lt  = f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_2/{str(imgnum).zfill(6)}.png"
        path_Rt  = f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_3/{str(imgnum).zfill(6)}.png"
        path_Lt1 = f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_2/{str(imgnum + 1).zfill(6)}.png"
        path_Rt1 = f"{self.cfg.odometry_home}{self.cfg.color_subdir}{seq}/image_3/{str(imgnum + 1).zfill(6)}.png"
        
        img_Lt  = cv2.imread(path_Lt)
        img_Rt  = cv2.imread(path_Rt)
        img_Lt1 = cv2.imread(path_Lt1)
        img_Rt1 = cv2.imread(path_Rt1)
        
        img_Lt  = cv2.cvtColor(img_Lt, cv2.COLOR_BGR2RGB)
        img_Rt  = cv2.cvtColor(img_Rt, cv2.COLOR_BGR2RGB)
        img_Lt1 = cv2.cvtColor(img_Lt1, cv2.COLOR_BGR2RGB)
        img_Rt1 = cv2.cvtColor(img_Rt1, cv2.COLOR_BGR2RGB)

        H, W, _ = img_Lt.shape
        crop_top = H % 32
        img_Lt  = img_Lt[crop_top:, :1216]
        img_Rt  = img_Rt[crop_top:, :1216]
        img_Lt1 = img_Lt1[crop_top:, :1216]
        img_Rt1 = img_Rt1[crop_top:, :1216]

        imgs = [torch.from_numpy(i).permute(2, 0, 1).float() / 255.0 for i in [img_Lt, img_Rt, img_Lt1, img_Rt1]]
        imgs_tensor = torch.stack(imgs)

        pose1_mat = torch.from_numpy(self.posesdict[seq][imgnum]).float()
        pose2_mat = torch.from_numpy(self.posesdict[seq][imgnum + 1]).float()

        # 행렬을 7차원 벡터로 변환
        vec1 = matrix_to_7vec(pose1_mat)
        vec2 = matrix_to_7vec(pose2_mat)

        # InitFromVec 사용 (이건 lietorch의 모든 버전에서 지원하는 표준 메서드입니다)
        se3_1 = SE3.InitFromVec(vec1.unsqueeze(0)) 
        se3_2 = SE3.InitFromVec(vec2.unsqueeze(0))

        rel_pose = se3_1.inv() * se3_2

        intrinsics = torch.from_numpy(self.intrinsicsdict[seq])

        return {
            'images': imgs_tensor,   # [4, 3, H, W]
            'rel_pose': rel_pose,    # SE3 객체 (batch_size=1)
            'intrinsics': intrinsics, # [4]
            'seq': seq,
            'imgnum': imgnum
        }
    
def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch])
    poses_vec = torch.cat([item['rel_pose'].vec() for item in batch], dim=0)
    rel_poses = SE3.InitFromVec(poses_vec)
    intrinsics = torch.stack([item['intrinsics'] for item in batch])
    seqs = [item['seq'] for item in batch]
    imgnums = [item['imgnum'] for item in batch]

    return {
        'images': images,
        'rel_poses': rel_poses,
        'intrinsics': intrinsics,
        'seqs': seqs,
        'imgnums': imgnums
    }
