import torch
import torch.nn as nn
from lietorch import SE3
from scipy.spatial.transform import Rotation as R
from .extractor import SuperPointExtractor
from utils.geo_utils import *
from .layer import *
from utils.DBA_utils import *
from joblib import Parallel, delayed

class VO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 기존 모듈들
        self.initializer = PoseInitializer()
        self.update_block = GraphUpdateBlock(input_dim=320, hidden_dim=256)
        self.solver = DBASolver()
        self.updater = PoseDepthUpdater()
        
        # 댐핑 파라미터
        # self.log_lmbda = nn.Parameter(torch.tensor(-4.6))
        self.log_lmbda = torch.tensor([2.3])
    

    def forward(self, batch, iters=4, mode='train'):
        device = next(self.parameters()).device
        if mode == 'train':
            for key in batch.keys():
                if torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(device)
                # 만약 tri_indices가 리스트 안에 텐서가 든 형태라면 (Collate 시)
                elif isinstance(batch[key], list):
                    batch[key] = [t.to(device) if torch.is_tensor(t) else t for t in batch[key]]
            kpts_t = batch['kpts']           # [B, 800, 2] (현재 프레임 특징점)
            pts_3d_t = batch['pts_3d']       # [B, 800, 3] (현재 프레임 3D)
            descs = batch['descs']           # [B, 800, 320]
            kpts_tp1 = batch['kpts_tp1']     # [B, 800, 2] (다음 프레임 대응점)
            tri_indices = batch['tri_indices'] # List of [T, 3] (삼각형 구조)
            mask = batch['mask']             # [B, 800] (유효 노드 마스크)
            intrinsics = batch['calib']      # [B, 4] (fx, fy, cx, cy)

            B, N, _ = kpts_t.shape
            
            R_init, tri_w, vp_conf, edge, edge_attr = self.initializer(descs, kpts_t, pts_3d_t, tri_indices, kpts_tp1, intrinsics)
   
            quats = matrix_to_quat(R_init) # [B, 4]
            trans = torch.zeros((B, 3), device=device)
            curr_pose = SE3.InitFromVec(torch.cat([trans, quats], dim=-1))
            curr_depth = pts_3d_t[..., 2:3] # [B, N, 1]
            h = torch.zeros((B, N, 256), device=device)

            predictions = []       


            for i in range(iters):
                pose_exp = SE3(curr_pose.data.unsqueeze(1).to(device))

                r, J_p, J_d = compute_geometry_data(
                    pts_3d_t, kpts_tp1, pose_exp, curr_depth, intrinsics
                )
                h, conf, a_p, a_d = self.update_block(
                    h, descs, r, vp_conf, vp_conf, edge, edge_attr, intrinsics, 
                    kpts_t, pts_3d_t
                )

                delta_pose, delta_depth = self.solver(r, conf, J_p, J_d, self.log_lmbda.exp().to(device))

                curr_pose = SE3.exp(a_p * delta_pose) * curr_pose
                curr_depth = curr_depth + a_d * delta_depth

                predictions.append({
                    'pose': curr_pose.matrix(), # [B, 4, 4]
                    'depth': curr_depth,
                    'conf': conf
                })
            return {
            'pose_matrices': [p['pose'] for p in predictions], # [B, 4, 4] 리스트
            'confidences': [p['conf'] for p in predictions]    # [B, N, 1] 리스트 (solver용 conf)
        }