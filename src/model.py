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
        self.corr_block = CorrBlock()
        self.cyclic_module = CyclicErrorModule(cfg.baseline)       
        self.update_block = UpdateBlock(hidden_dim=256)
        self.init_depth_net = nn.Linear(256, 1)
        self.DBA = DBASolver()
        self.DBA_Updater = PoseDepthUpdater()
        self.lmbda = nn.Parameter(torch.tensor(1e-4))

    def forward(self, batch, iters=8):
        device = self.lmbda.device
        calib = batch['calib'].to(device)
        B = calib.shape[0]

        # --- feature and descriptor ---
        if self.training:
            node_feats = batch['node_features'].to(device)  # [B, 3, 256+2(uv)]
            edges = batch['edges'].to(device) # [B, N]
            edge_attr = batch['edge_attr'].to(device) # []
            kpts_all = batch['kpts'].to(device)

            BN = node_feats.shape[0]
            D = node_feats.shape[-1]
            refined_desc_all, _ = self.GAT(node_feats.view(-1, D), edges, edge_attr)
            refined_desc_all = refined_desc_all.view(BN, -1, 256) 
        
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

        refined_desc_all = refined_desc_all.view(B, 4, -1, 256)
        kpts_all = kpts_all.view(B, 4, -1, 2)
        
        f_all = refined_desc_all 
        f_Lt = f_all[:, 0]
        f_others = f_all[:, 1:3] 
        kpts_Lt = kpts_all[:, 0]
        
        f_Lt_expand = f_Lt.unsqueeze(1).expand(-1, 2, -1, -1)
        combined_corr = self.corr_block(f_Lt_expand, f_others)
        
        c_stereo = combined_corr[:, 0]
        c_temp = combined_corr[:, 1]

        N = kpts_Lt.shape[1]
        curr_pose = SE3.Identity(B, device=device)
        curr_depth = torch.ones((B, N, 1), device=device) * 5.0
        h = torch.zeros((B, N, 256), device=device)

        poses_history = []
        weights_history = []
        errors_history = []

        for i in range(iters):
            e_proj = self.cyclic_module(kpts_Lt, curr_depth, curr_pose, calib)
            h, r, w, a_p, a_d = self.update_block(h, c_temp, c_stereo, e_proj, f_Lt, edges, edge_attr) 
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