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
        self.extractor = SuperPointExtractor()
        self.DT = compute_delaunay_edges

        self.stereo_matcher = EpipolarCrossAttention(feature_dim=256)
        self.temporal_matcher = TemporalCrossAttention(feature_dim=256)

        self.GAT = GeometricGAT(in_channels=258, out_channels=256)
        self.cyclic_module = CyclicErrorModule(cfg.baseline)       
        self.update_block = GraphUpdateBlock(hidden_dim=256)
        
        self.DBA = DBASolver()
        self.DBA_Updater = PoseDepthUpdater()
        self.log_lmbda = nn.Parameter(torch.tensor(-4.6))
        self.baseline = cfg.baseline
        self.focal = cfg.focal

    def forward(self, batch, iters=8, gt_guide=None):
        device = self.log_lmbda.device
        calib = batch['calib'].to(device)
        B = calib.shape[0]

        # --- feature and descriptor extraction ---
        if self.training:
            node_feats = batch['node_features'].to(device)
            edges_list = batch['edges']      
            edge_attr_list = batch['edge_attr'] 
            kpts_all = batch['kpts'].to(device)
            B_total, N, D = node_feats.shape 
        else:
            images = batch['images'].to(device)
            B, V, C, H, W = images.shape
            images_flat = images.view(B*V, C, H, W)
            kpts_all, desc_all = self.extractor(images_flat)
            
            all_edges = []
            all_edge_attrs = []
            kpts_np = kpts_all.detach().cpu().numpy()
            
            for i in range(B * V):
                edge_np = self.DT(kpts_np[i])
                edge_torch = torch.from_numpy(edge_np).to(device).long()
                src_pts = kpts_all[i][edge_torch[0]]
                dst_pts = kpts_all[i][edge_torch[1]]
                
                # [에피폴라 힌트] y축 차이가 적을수록 매칭 확률이 높음을 인코딩
                diff = src_pts - dst_pts 
                dist = torch.norm(diff, dim=1, keepdim=True)
                attr = torch.cat([dist, diff], dim=1) 
                
                all_edges.append(edge_torch)
                all_edge_attrs.append(attr)
            
            edges_list = all_edges
            edge_attr_list = all_edge_attrs
            size_tensor = torch.tensor([W, H], device=device).view(1, 1, 2)
            desc_ready = desc_all.transpose(1, 2) if desc_all.shape[1] == 256 else desc_all
            node_feats = torch.cat([desc_ready, kpts_all / size_tensor], dim=-1)
            B_total = B * V
            N = node_feats.shape[1]
            D = node_feats.shape[2]

        # --- GAT 기반 특징 정교화 (Epipolar-aware) ---
        flat_edges_list = []
        flat_attr_list = []
        for i in range(B_total):
            e = edges_list[i].to(device)
            flat_edges_list.append(e + i * N)
            
            # [에피폴라 힌트 주입] y축 차이가 0인 엣지에 가중치를 줌
            a = edge_attr_list[i].to(device)
            flat_attr_list.append(a)
        
        edges_combined = torch.cat(flat_edges_list, dim=1)
        edge_attr_combined = torch.cat(flat_attr_list, dim=0)

        refined_desc_all, _ = self.GAT(node_feats.reshape(-1, D), edges_combined, edge_attr_combined)
        refined_desc_all = refined_desc_all.view(B_total, N, -1)

        refined_desc_all = refined_desc_all.view(B, 4, -1, 256)
        kpts_all = kpts_all.view(B, 4, -1, 2)
        
        f_Lt, kpts_Lt = refined_desc_all[:, 0], kpts_all[:, 0]
        f_Rt, kpts_Rt = refined_desc_all[:, 1], kpts_all[:, 1]
        f_Lt1, kpts_Lt1 = refined_desc_all[:, 2], kpts_all[:, 2]

        # --- [2] Stereo Initialization (루프 진입 전 핵심!) ---
        # 5.0 고정값 대신 스테레오 매칭을 통해 실제 거리를 측정합니다.
        _, init_disp, _ = self.stereo_matcher(f_Lt, f_Rt, kpts_Lt, kpts_Rt)
        
        # d = (f * B) / disparity
        curr_depth = (self.focal * self.baseline) / (init_disp + 1e-6)
        curr_depth = torch.clamp(curr_depth, min=0.1, max=100.0) # 안전장치
        
        curr_pose = SE3.Identity(B, device=device)
        h = torch.zeros((B, kpts_Lt.shape[1], 256), device=device)

        # --- [3] Iteration 0: Initial Temporal Look ---
        # 루프 시작 전, 현재(정지) 가설로 Lt -> Lt1 오차를 먼저 확인합니다.
        kpts_t1_pred = project_kpts(kpts_Lt, curr_depth, curr_pose, calib)
        _, flow_res, conf_t = self.temporal_matcher(f_Lt, f_Lt1, kpts_t1_pred, kpts_Lt1, iter_idx=0)

        poses_history, weights_history, errors_history = [], [], []

        # --- [4] Iterative Update Loop ---
        for i in range(iters):
            if i == 0 and gt_guide is not None:
                curr_pose = SE3.InitFromVec(gt_guide)

            # 1. GAT + GRU 업데이트
            # flow_res(매칭 오차)와 엣지 정보를 GAT가 분석하여 GRU에 전달
            h, r, w, a_p, a_d = self.update_block(h, flow_res, f_Lt, edges_list, edge_attr_list) 
            
            # 2. 미분 기반 최적화 (DBA)
            J_p, J_d = compute_projection_jacobian(kpts_Lt, curr_depth, calib)
            lmbda = torch.exp(self.log_lmbda)
            delta_pose_dba, delta_depth_dba = self.DBA(r, w, J_p, J_d, lmbda)
            
            # 3. 포즈 및 깊이 업데이트
            curr_pose, curr_depth = self.DBA_Updater(curr_pose, curr_depth, delta_pose_dba, delta_depth_dba, a_p, a_d)

            # 4. Adaptive Temporal Matching (다음 루프를 위한 매칭 갱신)
            # 업데이트된 포즈로 다시 투영하고, 윈도우를 줄여가며 정밀 매칭 수행
            kpts_t1_pred = project_kpts(kpts_Lt, curr_depth, curr_pose, calib)
            _, flow_res, conf_t = self.temporal_matcher(f_Lt, f_Lt1, kpts_t1_pred, kpts_Lt1, iter_idx=i+1)

            poses_history.append(curr_pose)
            weights_history.append(w)
            errors_history.append(flow_res)

        return (SE3(torch.stack([p.data for p in poses_history])), torch.stack(weights_history), torch.stack(errors_history))
    
def project_kpts(kpts_t0, depth_t0, pose_t0t1, calib):
    """
    kpts_t0: [B, N, 2] - 현재 프레임의 2D 특징점 (u, v)
    depth_t0: [B, N, 1] - 현재 프레임의 깊이 (Z)
    pose_t0t1: lietorch.SE3 객체 - t0에서 t1으로의 상대 포즈 변화
    calib: [B, 4] - 카메라 내상수 [fx, fy, cx, cy]
    """
    B, N, _ = kpts_t0.shape
    device = kpts_t0.device
    
    fx, fy, cx, cy = calib[:, 0:1], calib[:, 1:2], calib[:, 2:3], calib[:, 3:4]

    # 1. 2D 픽셀 좌표 -> 3D 카메라 좌표계 (Back-projection)
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy
    u, v = kpts_t0[..., 0:1], kpts_t0[..., 1:2]
    x = (u - cx.unsqueeze(1)) * depth_t0 / fx.unsqueeze(1)
    y = (v - cy.unsqueeze(1)) * depth_t0 / fy.unsqueeze(1)
    z = depth_t0
    
    pts_3d_t0 = torch.cat([x, y, z], dim=-1) # [B, N, 3]

    # 2. SE3 포즈 변환 (t0 좌표계 -> t1 좌표계)
    # lietorch의 SE3 객체는 [B, N, 3] 형태의 포인트 클라우드 변환을 지원합니다.
    # pose_t0t1가 [B] 크기라면 노드 개수(N)만큼 확장해서 적용해야 할 수도 있습니다.
    pts_3d_t1 = pose_t0t1.unsqueeze(1) * pts_3d_t0 # [B, N, 3]

    # 3. 3D 카메라 좌표계 -> 2D 픽셀 좌표계 (Projection)
    # u' = x' * fx / z' + cx
    # v' = y' * fy / z' + cy
    x1, y1, z1 = pts_3d_t1[..., 0:1], pts_3d_t1[..., 1:2], pts_3d_t1[..., 2:3]
    
    # zero-division 방지
    z1 = torch.clamp(z1, min=0.1)
    
    u1 = (x1 * fx.unsqueeze(1) / z1) + cx.unsqueeze(1)
    v1 = (y1 * fy.unsqueeze(1) / z1) + cy.unsqueeze(1)
    
    kpts_t1_pred = torch.cat([u1, v1], dim=-1) # [B, N, 2]
    
    return kpts_t1_pred