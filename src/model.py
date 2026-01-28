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
        # 기존 모듈들
        self.initializer = GeoVOModel(sigma_voting=2.0) # SVD + Voting
        self.update_block = GraphUpdateBlock()
        self.solver = DBASolver()
        self.updater = PoseDepthUpdater()
        
        # 댐핑 파라미터
        self.log_lmbda = nn.Parameter(torch.tensor(-4.6)) 

    def forward(self, batch, iters=8):
        """
        batch: Dataset/Dataloader에서 전처리되어 넘어온 데이터
        """
        # [Step 0] 데이터 로드
        kpts_t = batch['kpts']           # [B, N, 2] (현재 t 프레임 특징점)
        pts_3d_t = batch['pts_3d']       # [B, N, 3] (현재 t 프레임 3D)
        descs = batch['descs']           # [B, N, 256]
        kpts_tp1 = batch['kpts_tp1']     # [B, N, 2] (다음 t+1 프레임 특징점 - 정답 역할)
        tri_indices = batch['tri_indices'] # List of [T, 3]
        mask = batch['mask']             # [B, N]
        intrinsics = batch['calib']      # [B, 4] (fx, fy, cx, cy)

        B, N, _ = kpts_t.shape
        device = kpts_t.device

        # --- [Step 1] GAT 실행 (이미지 문맥 및 기하 구조 파악) ---
        # 전처리된 kpts_t와 pts_3d_t를 사용하여 GAT 노드 피처 추출
        # node_feat: [B, N, 256]
        edges = tri_indices_to_edges(tri_indices, B, N, device)
        edge_attr = get_edge_attributes(edges, kpts_t, B, N)
        
        node_feat_flat, _ = self.initializer.gat(
            x=descs.view(-1, 256), 
            edge_index=edges, 
            edge_attr=edge_attr,
            kpts=kpts_t.view(-1, 2),
            pts_3d=pts_3d_t.view(-1, 3)
        )
        node_feat = node_feat_flat.view(B, N, 256) * mask.unsqueeze(-1)

        R_init, tri_weights, vp_conf = self.initializer(
            node_feat=node_feat,
            kpts=kpts_t,
            pts_3d=pts_3d_t,
            pts_3d_tp1=None, 
            tri_indices=tri_indices,
            focal=intrinsics[:, 0],
            cx=intrinsics[:, 2]
        )
        
        # Pose 초기화 (R은 보팅 결과, t는 0으로 시작)
        curr_pose = SE3.init(R_init, torch.zeros((B, 3), device=device))
        curr_depth = pts_3d_t[..., 2:3] # Z-depth
        
        # GRU Hidden State 초기화
        h = torch.zeros((B, N, 256), device=device)

        # --- [Step 3] GRU + DBA 루프 (Pose, Depth 수정) ---
        poses_list = []
        for i in range(iters):
            # 1. Residual 계산
            kpts_pred = self.projector(kpts_t, curr_depth, curr_pose, intrinsics)
            r = kpts_tp1 - kpts_pred 

            # 2. Update Block: 인자명 edge_index로 수정
            h, conf, a_p, a_d = self.update_block(
                h, node_feat, r, tri_weights, vp_conf, edge_index=edges
            )
            
            conf = conf * mask.unsqueeze(-1)

            # 3. Solver & Updater (기존 동일)
            J_p, J_d = compute_projection_jacobian(kpts_t, curr_depth, intrinsics)
            delta_pose, delta_depth = self.solver(r, conf, J_p, J_d, self.log_lmbda.exp())

            curr_pose, curr_depth = self.updater(
                curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d
            )
            
            poses_list.append(curr_pose)

        return {
            'poses': poses_list,      # Iteration별 Pose 결과 (학습용)
            'final_pose': curr_pose,  # 최종 추정 Pose
            'final_depth': curr_depth # 최종 수정된 Depth
        }