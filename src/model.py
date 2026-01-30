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
        self.log_lmbda = nn.Parameter(torch.tensor([0.0]))
        self.log_lmbda.data = torch.clamp(self.log_lmbda.data, min=0.0, max=5.0)
    

    def forward(self, batch, iters=4, mode='train'):
        if mode == 'train':
            device = self.log_lmbda.device # register_buffer 사용 시 편리함
            
            # 1. 데이터 입력 및 전처리
            kpts_t = batch['kpts'].to(device)
            pts_3d_t = batch['pts_3d'].to(device)
            descs = batch['descs'].to(device)
            kpts_tp1 = batch['kpts_tp1'].to(device)
            tri_indices = batch['tri_indices'] # 리스트 형태 유지
            intrinsics = batch['calib'].to(device)
            mask = batch['mask'].to(device).unsqueeze(-1).float() # (B, N, 1)

            kpts_t = kpts_t * mask
            pts_3d_t = pts_3d_t * mask
            kpts_tp1 = kpts_tp1 * mask
            descs = descs * mask

            B, N, _ = kpts_t.shape
        elif mode == 'eval':
            12
        
        # 2. 초기값 결정 (GAT + SVD)
        R_init, tri_w, vp_conf, edge, edge_attr = self.initializer(
            descs, kpts_t, pts_3d_t, tri_indices, kpts_tp1, intrinsics
        )
   
        # 초기 Pose & Depth 설정
        quats = matrix_to_quat(R_init) 
        trans = torch.zeros((B, 3), device=device)
        curr_pose = SE3.InitFromVec(torch.cat([trans, quats], dim=-1))
        curr_depth = pts_3d_t[..., 2:3]
        h = torch.zeros((B, N, 256), device=device)

        predictions = []       

        for i in range(iters):
            # 1. Geometry 데이터 계산 (여기서 r이 재투영 오차입니다)
            pose_for_act = SE3(curr_pose.data.unsqueeze(1))
            r, J_p, J_d = compute_geometry_data(
                pts_3d_t, kpts_tp1, pose_for_act, curr_depth, intrinsics
            )
            r = r * mask # 유효하지 않은 점 마스킹 (Loss 계산 시 매우 중요)
            
            # 2. GAT + GRU 업데이트
            h, conf, a_p, a_d = self.update_block(
                h * mask, descs, r, vp_conf, vp_conf, edge, edge_attr, intrinsics, 
                kpts_t, pts_3d_t
            )
            conf = conf * mask

            # 3. DBA Solver 실행
            delta_pose, delta_depth = self.solver(
                r, conf, J_p, J_d, self.log_lmbda.exp(), i # iters 대신 i 전달 권장
            )

            # 4. 업데이트
            curr_pose, curr_depth = self.updater(
                curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d, i # iters 대신 i 전달 권장
            )

            # [수정 포인트] 리스트에 r(residual)을 추가로 저장합니다.
            predictions.append({
                'pose': curr_pose.matrix(),
                'conf': conf,
                'res': r # <--- Loss에서 사용할 재투영 잔차
            })

        # [수정 포인트] 최종 반환값에 'residuals' 키를 추가합니다.
        return {
            'pose_matrices': [p['pose'] for p in predictions],
            'confidences': [p['conf'] for p in predictions],
            'residuals': [p['res'] for p in predictions] # <--- 추가된 부분
        }