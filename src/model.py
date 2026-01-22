import torch

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
        self.lmbda = nn.Parameter(torch.tensor(1e-3))

    def forward(self, batch, iters=8):
        device = self.lmbda.device
        calib = batch['calib'].to(device)
        B = calib.shape[0]

        # --- feature and descriptor ---
        if self.training:
            # node_feats shape: [B*4, 800, 258]
            node_feats = batch['node_features'].to(device)
            edges_list = batch['edges']      # List of [2, E_i]
            edge_attr_list = batch['edge_attr'] # List of [E_i, 3]
            kpts_all = batch['kpts'].to(device)

            B_total, N, D = node_feats.shape # B_total = B * 4

            # 1. 가변 길이 엣지 리스트를 GPU에서 통합 (오프셋 적용)
            flat_edges_list = []
            flat_attr_list = []
            for i in range(B_total):
                # 각 뷰/배치마다 노드 오프셋(i * 800)을 더함
                e = edges_list[i].to(device) if not isinstance(edges_list[i], torch.Tensor) else edges_list[i].to(device)
                flat_edges_list.append(e + i * N)
                
                a = edge_attr_list[i].to(device) if not isinstance(edge_attr_list[i], torch.Tensor) else edge_attr_list[i].to(device)
                flat_attr_list.append(a)
            
            # GAT 입력을 위한 통합 텐서
            edges_combined = torch.cat(flat_edges_list, dim=1)      # [2, Total_E]
            edge_attr_combined = torch.cat(flat_attr_list, dim=0)   # [Total_E, 3]

            # 2. GAT 연산
            # node_feats를 [Total_N, D]로 펼쳐서 전달
            refined_desc_all, _ = self.GAT(node_feats.reshape(-1, D), edges_combined, edge_attr_combined)
            
            # 3. 다시 원래 모양 [B*4, 800, 256]으로 복구
            refined_desc_all = refined_desc_all.view(B_total, N, -1)
            
            # UpdateBlock을 위해 edges 정보를 보존 (나중에 UpdateBlock 안에서 또 써야 하므로)
            edges = edges_list
            edge_attr = edge_attr_list
        
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
                edge_offset = edge_torch + (i * kpts_all.shape[1])
                
                src_pts = kpts_all[i][edge_torch[0]]
                dst_pts = kpts_all[i][edge_torch[1]]
                attr = torch.norm(src_pts - dst_pts, dim=1, keepdim=True)
                
                all_edges.append(edge_offset)
                all_edge_attrs.append(attr)
            
            edges = torch.cat(all_edges, dim=1)
            edge_attr = torch.cat(all_edge_attrs, dim=0)
            
            size_tensor = torch.tensor([W, H], device=device).view(1, 1, 2)
            node_feats = torch.cat([desc_all.transpose(1, 2), kpts_all / size_tensor], dim=-1)
            
            refined_desc_all, _ = self.GAT(node_feats.view(-1, 258), edges, edge_attr)
            refined_desc_all = refined_desc_all.view(B*V, -1, 256)

        # --- Iterative Update Loop 준비 ---
        refined_desc_all = refined_desc_all.view(B, 4, -1, 256)
        kpts_all = kpts_all.view(B, 4, -1, 2)
        
        f_all = refined_desc_all 
        f_Lt = f_all[:, 0]
        kpts_Lt = kpts_all[:, 0]
        

        N = kpts_Lt.shape[1]
        curr_pose = SE3.Identity(B, device=device)
        curr_depth = torch.ones((B, N, 1), device=device) * 5.0
        h = torch.zeros((B, N, 256), device=device)

        poses_history = []
        weights_history = []
        errors_history = []

        for i in range(iters):
            e_proj = self.cyclic_module(kpts_Lt, curr_depth, curr_pose, calib)
            
            h, r, w, a_p, a_d = self.update_block(h, e_proj, f_Lt, edges, edge_attr) 
            
            J_p, J_d = compute_projection_jacobian(kpts_Lt, curr_depth, calib)
            delta_pose_dba, delta_depth_dba = self.DBA(r, w, J_p, J_d, self.lmbda)
            curr_pose, curr_depth = self.DBA_Updater(curr_pose, curr_depth, delta_pose_dba, delta_depth_dba, a_p, a_d)

            poses_history.append(curr_pose)
            weights_history.append(w)
            errors_history.append(e_proj)

        poses_h = SE3(torch.stack([p.data for p in poses_history]))
        weights_h = torch.stack(weights_history) 
        errors_h = torch.stack(errors_history) 
        return (poses_h, weights_h, errors_h)