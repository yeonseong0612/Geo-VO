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
        self.GAT = GeometricGAT(in_channels=258, out_channels=256)
        self.cyclic_module = CyclicErrorModule(cfg.baseline)       
        self.update_block = GraphUpdateBlock(hidden_dim=256)
        self.init_depth_net = nn.Linear(256, 1)
        self.DBA = DBASolver()
        self.DBA_Updater = PoseDepthUpdater()
        self.log_lmbda = nn.Parameter(torch.tensor(-4.6))

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

        # --- Iterative Update Loop 준비 ---
        refined_desc_all = refined_desc_all.view(B, 4, -1, 256)
        kpts_all = kpts_all.view(B, 4, -1, 2)
        
        f_Lt = refined_desc_all[:, 0]
        kpts_Lt = kpts_all[:, 0]
        
        curr_pose = SE3.Identity(B, device=device)
        curr_depth = torch.ones((B, N, 1), device=device) * 5.0
        h = torch.zeros((B, N, 256), device=device)

        poses_history = []
        weights_history = []
        errors_history = []

        for i in range(iters):
            # -------------------------------------------------------
            # [수정] Iteration 0에서 GT 가이드(Teacher Forcing) 주입
            # -------------------------------------------------------
            if i == 0 and gt_guide is not None:
                curr_pose = SE3.InitFromVec(gt_guide)
            # -------------------------------------------------------

            # 1. 4장 이미지 순환 투영 에러 계산
            e_proj = self.cyclic_module(kpts_Lt, curr_depth, curr_pose, calib)
            
            # 2. 업데이트 블록 (GAT + GRU)
            # h: hidden state, r: residual, w: confidence weight
            h, r, w, a_p, a_d = self.update_block(h, e_proj, f_Lt, edges_list, edge_attr_list) 
            
            # 3. 미분 기반 최적화 (DBA)
            J_p, J_d = compute_projection_jacobian(kpts_Lt, curr_depth, calib)
            lmbda = torch.exp(self.log_lmbda)
            
            delta_pose_dba, delta_depth_dba = self.DBA(r, w, J_p, J_d, lmbda)
            
            # 4. 포즈 및 깊이 업데이트 (학습된 보폭 alpha 반영)
            curr_pose, curr_depth = self.DBA_Updater(curr_pose, curr_depth, delta_pose_dba, delta_depth_dba, a_p, a_d)

            poses_history.append(curr_pose)
            weights_history.append(w)
            errors_history.append(e_proj)

        # SE3 객체 형태로 패킹하여 반환
        poses_h = SE3(torch.stack([p.data for p in poses_history]))
        weights_h = torch.stack(weights_history) 
        errors_h = torch.stack(errors_history) 
        
        return (poses_h, weights_h, errors_h)