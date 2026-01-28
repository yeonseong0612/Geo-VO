import torch
import torch.nn as nn
from lietorch import SE3
from .extractor import SuperPointExtractor
from utils.geo_utils import *
from .layer import *
from utils.DBA_utils import *
from joblib import Parallel, delayed

class VO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.extractor = SuperPointExtractor()
        self.selector = DescSelector()
        self.DT = compute_delaunay_edges  # Delaunay Triangulation

        self.log_lmbda = nn.Parameter(torch.tensor(-4.6)) # 초기 댐핑 lmbda = 0.01
        self.initializer = GeoVOModel(sigma_voting=2.0) # SVD + Voting 모듈
        self.baseline = cfg.baseline
        self.update_block = GraphUpdateBlock()
        self.solver = DBASolver()
        self.updater = PoseDepthUpdater()

    def projector(self, kpts, depth, pose, intrinsics):
        """
        kpts: [B, N, 2]
        depth: [B, N, 1] 또는 [B, N]
        pose: SE3 객체 [B]
        intrinsics: [B, 4]
        """
        B, N, _ = kpts.shape
        if depth.dim() == 3:
            depth = depth.squeeze(-1) # [B, N]

        fx = intrinsics[:, 0:1] 
        fy = intrinsics[:, 1:2]
        cx = intrinsics[:, 2:3]
        cy = intrinsics[:, 3:4]

        # 1. 2D -> 3D
        x = (kpts[..., 0] - cx) * depth / fx
        y = (kpts[..., 1] - cy) * depth / fy
        z = depth
        pts_3d = torch.stack([x, y, z], dim=-1) # [B, N, 3]

        # 2. 3D 공간에서 Pose 변환
        # [수정] (B, 1)을 튜플로 묶어서 전달합니다.
        pts_3d_transformed = pose.view((B, 1)) * pts_3d # [B, N, 3]

        # 3. 3D -> 2D
        z_next = pts_3d_transformed[..., 2] + 1e-8
        u_next = fx * (pts_3d_transformed[..., 0] / z_next) + cx
        v_next = fy * (pts_3d_transformed[..., 1] / z_next) + cy
        
        return torch.stack([u_next, v_next], dim=-1) # [B, N, 2]
    
    def forward(self, batch, iters=8):
        # 1. 데이터 로더에서 사전 추출된 값들 가져오기
        # batch['kpts']: [B, 800, 2]
        # batch['pts_3d']: [B, 800, 3] (사전 계산된 3D 점)
        # batch['tri_indices']: [B, T, 3] (사전 계산된 삼각형)
        # batch['descs']: [B, 800, 256]
        
        kpts = batch['kpts']
        pts_3d_tm1 = batch['pts_3d'] # 이전 프레임 기준 3D
        tri_indices = batch['tri_indices']
        descs = batch['descs']
        intrinsics = batch['calib']

        # 2. Geo-VO 초기화 (SVD + Voting)
        # 학습 시에는 pts_3d_t(현재 3D)도 사전 계산되어 있을 것이므로 이를 활용
        R_init, tri_weights, vp_conf = self.initializer(
            descs, kpts, pts_3d_tm1, tri_indices, intrinsics
        )
        
        # Pose 초기화 (R은 보팅 결과, t는 0)
        curr_pose = SE3.init(R_init, torch.zeros((R_init.shape[0], 3), device=kpts.device))
        curr_depth = pts_3d_tm1[..., 2:3] # 초기 깊이 (Z)

        # 3. Context Feature 준비 (GAT 노드 피처)
        # 이미 사전 추출된 descs와 kpts를 GAT에 태워 context(h) 생성
        node_feat, _ = self.gat(descs, kpts, pts_3d_tm1, edges_precomputed)
        h = torch.zeros_like(node_feat) # GRU hidden state

        # 4. Iterative DBA 루프
        poses_list = []
        for i in range(iters):
            # (A) 현재 추정 Pose로 재투영 수행하여 오차(r) 계산
            # pts_3d_tm1을 curr_pose로 변환 후 2D로 투영
            kpts_pred = self.projector(kpts, curr_depth, curr_pose, intrinsics)
            r = batch['target_kpts'] - kpts_pred # 관측된 2D(Next Frame)와의 차이

            # (B) 업데이트 블록 (가중치 & 보폭 산출)
            # 사전 추출된 tri_weights와 vp_conf가 여기서 '가이드' 역할을 함
            h, conf, a_p, a_d = self.update_block(
                h, node_feat, r, tri_weights, vp_conf, edges_precomputed
            )

            # (C) DBA 최적화 및 Pose 갱신
            J_p, J_d = compute_projection_jacobian(kpts, curr_depth, intrinsics)
            delta_pose, delta_depth = self.solver(r, conf, J_p, J_d, lmbda=0.01)
            
            curr_pose, curr_depth = self.updater(
                curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d
            )

            poses_list.append(curr_pose)

        return {'poses': poses_list}