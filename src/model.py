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
    
    def forward(self, batch, iters=8, mode='train'):
        B = batch['calib'].shape[0]
        device = batch['calib'].device
        intrinsics = batch['calib']
        V = 4 # View 개수 명시

        # --- [Step 1: Raw Feature 획득] ---
        if 'raw_node_features' in batch:
            # [B, 4, 800, 256] -> [B*4, 800, 256]
            f_all_raw = batch['raw_node_features'].view(B * V, 800, 256)
            k_all_raw = batch['raw_kpts'].view(B * V, 800, 2)
        else:
            imgs_stacked = batch['imgs'].view(B * V, 3, 352, 1216)
            k_all_raw, f_all_raw = self.extractor(imgs_stacked)

        f_all, k_all, _ = self.selector(k_all_raw, f_all_raw, (352, 1216), top_k=128)

        N = k_all.shape[1] 
        D = f_all.shape[-1]

        k_split = k_all.view(B, V, N, 2)
        f_split = f_all.view(B, V, N, D)
        
        k_Lt, k_Rt, k_Lt1, k_Rt1 = [k_split[:, i] for i in range(V)]
        f_Lt, f_Rt, f_Lt1, f_Rt1 = [f_split[:, i] for i in range(V)]

        all_k_np = [k_split[b, i].detach().cpu().numpy() for b in range(B) for i in range(V)]        
        results = Parallel(n_jobs=6, backend="threading")(
            delayed(process_single_view)(k, self.DT) for k in all_k_np
        )
        edges_Lt, edge_attr_Lt = [], []
        for idx, (e_np, attr_np) in enumerate(results):
            if idx % 4 == 0: # 오직 Lt 이미지의 그래프만 추출
                edges_Lt.append(torch.from_numpy(e_np).long().to(device))
                edge_attr_Lt.append(torch.from_numpy(attr_np).float().to(device))
            
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