import torch
import torch.nn as nn
from lietorch import SE3
from .extractor import SuperPointExtractor
from utils.geo_utils import *
from .layer import *
from utils.DBA_utils import *

class VO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.extractor = SuperPointExtractor()
        self.DT = compute_delaunay_edges  # Delaunay Triangulation

        # 1. 시각 및 기하 매칭 모듈
        self.stereo_matcher = EpipolarCrossAttention(feature_dim=256)
        self.temporal_matcher = TemporalCrossAttention(q_dim=544, kv_dim=256)

        # 2. 깊이 초기화 및 특징 융합 (우리가 설계한 핵심!)
        self.stereo_depth_mod = StereoDepthModule(feature_dim=256)
        self.geo_bottleneck = GeometricBottleneck(visual_dim=768, geo_dim=64, out_dim=256)

        # 3. 그래프 최적화 및 업데이트 블록
        # Bottleneck을 거쳐 256차원으로 고정된 입력을 받습니다.
        self.update_block = GraphUpdateBlock(hidden_dim=256)

        # 4. 수치적 솔버
        self.DBA = DBASolver()
        self.DBA_Updater = PoseDepthUpdater(min_depth=0.1, max_depth=100.0)
        self.log_lmbda = nn.Parameter(torch.tensor(-4.6)) # 초기 댐핑 lmbda = 0.01
        
        self.baseline = cfg.baseline

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
        """
        batch 구성 (from vo_collate_fn):
        - node_features: [B, 4, 800, 256] (Lt, Rt, Lt1, Rt1)
        - kpts: [B, 4, 800, 2]
        - edges: [B*4] 리스트 (각 원소는 [2, E_i])
        - edge_attr: [B*4] 리스트 (각 원소는 [E_i, 3])
        - calib: [B, 4] (fx, fy, cx, cy)
        """
        device = batch['node_features'].device
        B = batch['node_features'].shape[0]         # batch size
        N = 800                                     # keypoint number 

        # kp & desc(Lt, Rt, Lt1, Rt1)
        f_Lt, f_Rt, f_Lt1, f_Rt1 = [batch['node_features'][:, i] for i in range(4)] # [B, 800, 256]
        k_Lt, k_Rt, k_Lt1, k_Rt1 = [batch['kpts'][:, i] for i in range(4)]           # [B, 800, 2]\
        intrinsics = batch['calib'] # [B, 4]

        edges_Lt = []
        edge_attr_Lt = []   

        for b in range(B):
            idx = b * 4
            edges_Lt.append(batch['edges'][idx].to(device))
            edge_attr_Lt.append(batch['edge_attr'][idx].to(device))

        v_stereo, init_disp, conf_stereo, _ = self.stereo_matcher(f_Lt, f_Rt, k_Lt, k_Rt)

        v_stereo_feat, initial_depth = self.stereo_depth_mod(
            f_Lt, v_stereo, init_disp, conf_stereo, intrinsics, self.baseline
        )

        current_pose = SE3.Identity(B, device=device)
        kpts_t1_pred = self.projector(k_Lt, initial_depth, current_pose, intrinsics)

        v_temp_feat, geo_temp, flow_res, conf_temp = self.temporal_matcher(
            nodes_t=v_stereo_feat,     
            nodes_t1=f_Lt1,            
            kpts_t1_pred=kpts_t1_pred, 
            kpts_t1_actual=k_Lt1,      
            iter_idx=0                 
        )

        v_feat = torch.cat([v_stereo_feat[:, :, :512], v_temp_feat], dim=-1) # [B, N, 768]
        g_feat = torch.cat([v_stereo_feat[:, :, 512:], geo_temp], dim=-1) 

        h = self.geo_bottleneck(v_feat, g_feat) 
        poses_list = []
        depths_list = []
        weights_list = []   

        cur_h = h
        cur_pose = current_pose
        cur_depth = initial_depth

        kpts_t1_actual_matched = kpts_t1_pred + flow_res    

        for i in range(iters):
            cur_h, r, w, a_p, a_d = self.update_block(
                cur_h, flow_res, v_feat, edges_Lt, edge_attr_Lt
            )

            J_p, J_d = compute_projection_jacobian(k_Lt, cur_depth, intrinsics)
            
            lmbda = self.log_lmbda.exp()
            delta_pose, delta_depth = self.DBA(flow_res + r, w, J_p, J_d, lmbda)

            cur_pose, cur_depth = self.DBA_Updater(
                cur_pose, cur_depth, delta_pose, delta_depth, a_p, a_d
            )

            kpts_t1_pred = self.projector(k_Lt, cur_depth, cur_pose, intrinsics)

            flow_res = kpts_t1_actual_matched - kpts_t1_pred

            poses_list.append(cur_pose)
            depths_list.append(cur_depth)
            weights_list.append(w)
        return {
            'poses': poses_list,
            'depths': depths_list,
            'weights': weights_list
        }   