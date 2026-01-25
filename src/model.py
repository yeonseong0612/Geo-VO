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
    
    def forward(self, batch, iters=8, mode='train'):
        """
        batch 구성 (from vo_collate_fn):
        - node_features: [B, 4, 800, 256] (Lt, Rt, Lt1, Rt1)
        - kpts: [B, 4, 800, 2]
        - edges: [B*4] 리스트 (각 원소는 [2, E_i])
        - edge_attr: [B*4] 리스트 (각 원소는 [E_i, 3])
        - calib: [B, 4] (fx, fy, cx, cy)
        """

        if mode == 'train':
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
        else: # Inference Mode (Raw Images)
            image = batch['imgs'] # [B, 4, 3, 376, 1241]
            B = image.shape[0]
            device = image.device
            intrinsics = batch['calib']

            # 1. 특징점 및 디스크립터 추출 (Lt, Rt, Lt1, Rt1)
            # 학습 시 N=800이었으므로 추론 시에도 상위 800개를 유지하는 것이 안전합니다.
            def extract_800(img):
                k, f = self.extractor(img)
                if k.shape[1] > 800:
                    k, f = k[:, :800], f[:, :800]
                elif k.shape[1] < 800:
                    pad = 800 - k.shape[1]
                    k = torch.cat([k, torch.zeros((B, pad, 2), device=device)], dim=1)
                    f = torch.cat([f, torch.zeros((B, pad, 256), device=device)], dim=1)
                return k, f

            k_Lt, f_Lt   = extract_800(image[:, 0])
            k_Rt, f_Rt   = extract_800(image[:, 1])
            k_Lt1, f_Lt1 = extract_800(image[:, 2])
            k_Rt1, f_Rt1 = extract_800(image[:, 3])

            edges_Lt, edge_attr_Lt = [], []
            kpts_all = [k_Lt, k_Rt, k_Lt1, k_Rt1]
            
            # 2. 리스트 구조 동기화 (가장 중요!)
            for b in range(B):
                for i in range(4): # 반드시 4번 돌아서 edges와 edge_attr의 개수를 맞춥니다.
                    k_np = kpts_all[i][b].detach().cpu().numpy()
                    e_np = self.DT(k_np) # [2, E] (2xE 형태 반환 확인)
                    
                    if e_np.shape[1] > 0:
                        # edge_attr 계산 ([dist, dx, dy])
                        src_pts = k_np[e_np[0]]
                        dst_pts = k_np[e_np[1]]
                        diff = src_pts - dst_pts
                        dist = np.linalg.norm(diff, axis=1, keepdims=True)
                        attr_np = np.concatenate([dist, diff], axis=1).astype(np.float32)
                        
                        # [VITAL FIX] 인덱스와 속성을 동시에 append 하여 1:1 매칭 보장
                        edges_Lt.append(torch.from_numpy(e_np).long().to(device))
                        edge_attr_Lt.append(torch.from_numpy(attr_np).float().to(device))
                    else:
                        # 에지가 없는 경우 (더미 생성으로 리스트 인덱스 유지)
                        edges_Lt.append(torch.zeros((2, 0), dtype=torch.long, device=device))
                        edge_attr_Lt.append(torch.zeros((0, 3), dtype=torch.float, device=device))

                
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