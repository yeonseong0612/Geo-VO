import torch
import torch.nn as nn
import torch.nn.functional as F
from lietorch import SE3
from torch_geometric.nn import GATv2Conv

import torch
import torch.nn as nn

from utils.geo_utils import *

class GeometricGAT(nn.Module):
    def __init__(self, in_channels, hidden_dim=256, heads=4, pos_dim=3):
        super().__init__()
        # 1. [u, v, Depth] Encoder
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )

        # 2. Residual Projection
        self.res_proj = nn.Linear(in_channels, hidden_dim)

        # 3. GATv2 (in_channel = Edge [Δu, Δv, dist])
        self.conv = GATv2Conv(
            in_channels, 
            hidden_dim // heads,
            heads=heads, 
            edge_dim=3,
            add_self_loops=False
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Linear(hidden_dim, 256)
        self.SiLU = nn.SiLU()

    def forward(self, x, edge_index, kpts, pts_3d, edge_attr=None):
        """
        x: [N, 256] - desc
        edge_index: [2, E] - tri_indices
        kpts: [N, 2] - (u, v)
        pts_3d: [N, 3] : (X, Y, Z)
        edge_attr: [E, 3] : [Δu, Δv, dist]
        """
        # [Step 1] Node = Desc + geo_info
        # Norm(1216, 352)
        norm_uv = kpts / torch.tensor([1216.0, 352.0], device=kpts.device)
        depth = pts_3d[:, 2:3] # Z Value
        
        # [N, 3] -> [N, 64]
        pos_feat = self.pos_encoder(torch.cat([norm_uv, depth], dim=-1))
        
        # [N, 256] + [N, 64] = [N, 320]
        x = torch.cat([x, pos_feat], dim=-1)

        # [Step 2] Edge_info : [Δu, Δv, dist]
        if edge_attr is None and edge_index is not None:
            src, dst = edge_index[0], edge_index[1]
            rel_uv = norm_uv[dst] - norm_uv[src]            # (Δu, Δv)
            dist = torch.norm(rel_uv, dim=-1, keepdim=True) # dist
            edge_attr = torch.cat([rel_uv, dist], dim=-1)   # [E, 3]

        # [Step 3] Identity for Residual
        identity = self.res_proj(x)

        # [Step 4] GATv2
        out, (edge_index_out, alpha) = self.conv(x, edge_index, edge_attr, return_attention_weights=True)
        
        # [Step 5] Post-processing & Residual Connection
        out = self.norm(out)
        out = self.SiLU(out)
        out = out + identity 
        
        out = self.projector(out)
        
        return out, alpha, edge_attr

class TriangleHead(nn.Module):
    def __init__(self, node_dim=256, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Weight(0~1) : Confidence of Triangle
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Normal Vector
        self.normal_head = nn.Linear(hidden_dim, 3) 

    def forward(self, node_feat, tri_indices):
        """
        node_feat: [B, N, 256] - GAT updateed node
        tri_indices: List of [T, 3]
        """
        B = node_feat.shape[0]
        all_weights = []
        all_normals = []

        for b in range(B):
            # 1. Make Trianle Feature[T, 3, 256]
            f1 = node_feat[b, tri_indices[b][:, 0]]
            f2 = node_feat[b, tri_indices[b][:, 1]]
            f3 = node_feat[b, tri_indices[b][:, 2]]

            # 2. Concat Feature[T, 768]
            f_tri = torch.cat([f1, f2, f3], dim=-1)

            # 3. MLP
            feat = self.mlp(f_tri)

            # 4. result
            weights = self.weight_head(feat)    #[T, 1]
            normals = torch.norm(self.normal_head(feat), p=2, dim=-1, keepdim=True) # [T, 3]

            all_weights.append(weights)
            all_normals.append(normals)
        return all_weights, all_normals
        
class PoseInitializer(nn.Module):
    def __init__(self, in_channels=320, node_dim=256):
        super().__init__()
        self.gat = GeometricGAT(in_channels=in_channels, hidden_dim=256)
        self.tri_head = TriangleHead(node_dim=node_dim)

    def forward(self, descs, kpts, pts_3d, tri_indices, kpts_tp1, intrinsics):
        B, N, _ = descs.shape
        device = descs.device
        # 1. GAT : feuture update
        edges = tri_indices_to_edges(tri_indices, B, N, device)
        node_feat_flat, _, edge_attr = self.gat(
            x=descs.view(-1, 256),
            edge_index=edges,
            kpts=kpts.view(-1, 2),
            pts_3d=pts_3d.view(-1, 3)
        )
        node_feat = node_feat_flat.view(B, N, 256)

        # 2. Tringle Weights
        weights_list, _ = self.tri_head(node_feat, tri_indices)

        final_R_list = []
        final_tri_weights = []
        final_vp_conf = []

        # 3. 배치 루프: 각 배치별 초기 Pose 결정
        for b in range(B):
            tris = tri_indices[b]
            w_j = weights_list[b]

            fx, fy, cx, cy = intrinsics[b]
            ux = (kpts_tp1[b, :, 0] - cx) / fx
            uy = (kpts_tp1[b, :, 1] - cy) / fy
            p_tp1_norm = torch.stack([ux, uy, torch.ones_like(ux)], dim=-1) # [N, 3]    

            K_j = compute_individual_Kj(tris, pts_3d[b], p_tp1_norm)
            R_j_candidates = batch_svd(K_j)

            r13 = R_j_candidates[:, 0, 2]
            r33 = R_j_candidates[:, 2, 2] + 1e-8
            xv_j = fx * (r13 / r33) + cx
            
            xv_star = differentiable_voting(xv_j, w_j, sigma=2.0)

            dist_sq = (xv_j - xv_star)**2
            s_j = torch.exp(-dist_sq / (2 * 2.0**2)).unsqueeze(-1) # [T, 1]

            combined_weights = w_j * s_j
            R_init = estimate_rotation_svd_differentiable(
                combined_weights, tris, pts_3d[b], p_tp1_norm
            )

            node_v_conf = torch.zeros((N, 1), device=device)
            node_v_conf.scatter_add_(0, tris.view(-1, 1).expand(-1, 1), s_j.repeat_interleave(3, dim=0))
            node_v_conf = torch.tanh(node_v_conf)

            final_R_list.append(R_init)
            final_tri_weights.append(combined_weights)
            final_vp_conf.append(node_v_conf)
        return torch.stack(final_R_list), final_tri_weights, torch.stack(final_vp_conf), edges, edge_attr


class DBASolver(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, r, w, J_p, J_d, lmbda):
        # r: [B, N, 2], w: [B, N, ?], J_p: [B, N, 2, 6], J_d: [B, N, 2, 1]
        B, N, _ = r.shape
        device = r.device
        
        # 1. w 차원 방어 로직 (중요!)
        # w가 1채널이면 댐핑을 0으로, 2채널이면 2번째 채널 사용
        conf = w[..., 0:1].unsqueeze(-1)    # [B, N, 1, 1]
        if w.shape[-1] >= 2:
            node_lambda = w[..., 1:2]       # [B, N, 1]
        else:
            node_lambda = torch.zeros((B, N, 1), device=device)

        # Hessian & Gradient 계산
        H_pp = torch.matmul(J_p.transpose(-1, -2), conf * J_p).sum(dim=1) # [B, 6, 6]
        H_pd = torch.matmul(J_p.transpose(-1, -2), conf * J_d)           # [B, N, 6, 1]
        H_dd = torch.matmul(J_d.transpose(-1, -2), conf * J_d).squeeze(-1) # [B, N, 1]

        g_p = torch.matmul(J_p.transpose(-1, -2), conf * r.unsqueeze(-1)).sum(dim=1) 
        g_d = torch.matmul(J_d.transpose(-1, -2), conf * r.unsqueeze(-1)).squeeze(-1) 

        # 2. Levenberg-Marquardt Damping 적용 (에러 수정 지점)
        # lmbda가 스칼라이므로 그냥 더해줘도 브로드캐스팅 됩니다.
        # 또는 lmbda.item()을 사용하거나 lmbda 자체를 활용합니다.
        diag_mask = torch.eye(6, device=device).unsqueeze(0) # [1, 6, 6]
        H_pp = H_pp + (lmbda * diag_mask) 
        
        # H_dd: [B, N, 1], lmbda: 스칼라, node_lambda: [B, N, 1]
        H_dd = H_dd + lmbda + node_lambda

        # 3. Schur Complement 및 inv_H_dd 계산
        inv_H_dd = 1.0 / (H_dd + 1e-2)
        H_pd_invHdd = H_pd * inv_H_dd.unsqueeze(-1) # [B, N, 6, 1]

        # H_eff = H_pp - sum(H_pd * inv_H_dd * J_pd^T)
        H_eff = H_pp - torch.matmul(H_pd_invHdd, H_pd.transpose(-1, -2)).sum(dim=1)
        H_eff = 0.5 * (H_eff + H_eff.transpose(-1, -2)) # 대칭성 강제 보장
        
        # g_eff = g_p - sum(H_pd * inv_H_dd * g_d)
        g_eff = g_p - (H_pd_invHdd * g_d.unsqueeze(-1)).sum(dim=1) # [B, 6, 1]

        # 4. 선형 시스템 풀기 (H_eff * delta_pose = g_eff)
        eps_ridge = 1e-3
        H_eff = H_eff + eps_ridge * torch.eye(6, device=device).unsqueeze(0)
        
        try:
            delta_pose = torch.linalg.solve(H_eff, g_eff) # [B, 6, 1]
        except RuntimeError: # linalg error 처리
            delta_pose = torch.zeros_like(g_eff)

        # 5. 최종 delta_depth 계산
        # v = H_pd^T * delta_pose
        # H_pd.transpose(-1, -2): [B, N, 1, 6], delta_pose: [B, 6, 1]
        # v: [B, N, 1, 1] -> squeeze -> [B, N, 1]
        v = torch.matmul(H_pd.transpose(-1, -2), delta_pose.unsqueeze(1)).squeeze(-1)
        delta_depth = inv_H_dd * (g_d - v)
        delta_pose = delta_pose.squeeze(-1)
        # 1프레임당 이동 2m, 회전 0.1rad(약 5도)로 제한
        delta_pose = torch.clamp(delta_pose, min=-2.0, max=2.0) 
        return delta_pose, delta_depth
    
class PoseDepthUpdater(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=100.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def forward(self, curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d):
        # 1. Depth Update: 보폭(a_d) 적용 및 강력한 범위 제한
        # delta_depth가 너무 클 경우를 대비해 한번 더 clamp 해주는 것이 안전합니다.
        safe_delta_d = torch.clamp(delta_depth, min=-5.0, max=5.0) 
        new_depth = curr_depth + a_d * safe_delta_d
        
        # 깊이가 음수가 되거나 무한히 멀어지는 것을 방지
        new_depth = torch.clamp(new_depth, min=self.min_depth, max=self.max_depth)

        # 2. Pose Update: 보폭(a_p) 적용
        # a_p는 우리가 GraphUpdateBlock에서 0.1 스케일로 줄였으므로 안정적입니다.
        scaled_delta = a_p * delta_pose

        # 리 군(Lie Group) 지수 사상 적용
        delta_SE3 = SE3.exp(scaled_delta)
        
        # T_new = Delta * T_curr (Local frame update)
        # 관례적으로 World-to-Camera 좌표계라면 이 순서가 맞습니다.
        new_pose = curr_pose * delta_SE3 

        return new_pose, new_depth
    

class GraphUpdateBlock(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        # feature(256) + residual(2) + tri_w(1) + vp_s(1) = 260
        fused_dim = input_dim + 4 

        self.spatial_gat = GeometricGAT(in_channels=fused_dim, hidden_dim=hidden_dim)
        self.norm_gat = nn.LayerNorm(hidden_dim)
        
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        
        # 가중치 헤드: [B, N, 2] -> (Confidence, Damping)
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(),
            nn.Linear(128, 2) 
        )
        
        # 포즈/깊이 업데이트 보폭 (Learnable Step Size)
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU(),
            nn.Linear(64, 1) # Sigmoid 제거 후 루프 내에서 처리
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, h, node_feat, r, tri_w, vp_s, edges, edge_attr, intrinsics, kpts, pts_3d):
        B, N, _ = node_feat.shape
        
        r_norm = r / intrinsics[:, :2].unsqueeze(1) 
        x_fused = torch.cat([node_feat, r_norm, tri_w, vp_s], dim=-1) # [B, N, 260]

        x_spatial_flat, _, _ = self.spatial_gat(
            x=x_fused.view(-1, x_fused.size(-1)), 
            edge_index=edges, 
            edge_attr=edge_attr,
            kpts=kpts.view(-1, 2),
            pts_3d=pts_3d.view(-1, 3)
        )
        x_spatial = self.norm_gat(x_spatial_flat.view(B, N, -1))

        h_flat = h.view(-1, self.hidden_dim)
        h_new_flat = self.gru(x_spatial.view(-1, self.hidden_dim), h_flat)
        h_new = self.norm_h(h_new_flat.view(B, N, -1))

        conf = torch.sigmoid(self.weight_head(h_new)[..., 0:1]) 
        
        a_p = torch.sigmoid(self.alpha_pose_head(h_new)).mean(dim=1) * 0.1
        a_d = torch.sigmoid(self.alpha_depth_head(h_new)) * 0.1
        
        return h_new, conf, a_p, a_d
    







class DescSelector(nn.Module):
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256), 
            nn.SiLU(),
            nn.Linear(256, out_dim)
        )
        self.score_head = nn.Linear(out_dim, 1)
        self.out_dim = out_dim

        self.init_weights()

    def init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.score_head.weight, 0.001)
        nn.init.constant_(self.score_head.bias, 0.0)
        

    def forward(self, kpts, desc, img_shape, top_k=128):
        """
        kpts: [B, N, 2], desc: [B, N, 256]
        img_shape: (H, W)
        """
        B, N, _ = kpts.shape
        H, W = img_shape
        device = kpts.device

        feat = self.mlp(desc) 
        scores = self.score_head(feat).squeeze(-1)

        x = kpts[..., 0]
        y = kpts[..., 1]

        mid_mask = (y > 0.2 * H) & (y <= 0.5 * H)
        bottom_mask = (y > 0.5 * H)
        
        mid_grid_x = (x / W * 8).long().clamp(0, 7)
        mid_grid_y = ((y - 0.2*H) / (0.3*H) * 4).long().clamp(0, 3)
        mid_grid_id = mid_grid_y * 8 + mid_grid_x 
        
        btm_grid_x = (x / W * 16).long().clamp(0, 15)
        btm_grid_y = ((y - 0.5*H) / (0.5*H) * 6).long().clamp(0, 5)
        btm_grid_id = 32 + (btm_grid_y * 16 + btm_grid_x) 

        grid_ids = torch.full((B, N), -1, dtype=torch.long, device=device)
        grid_ids[mid_mask] = mid_grid_id[mid_mask]
        grid_ids[bottom_mask] = btm_grid_id[bottom_mask]


        final_indices = []
        for b in range(B):
            grid_max_scores = torch.full((128,), -1e9, device=device)
            grid_max_idx = torch.full((128,), -1, dtype=torch.long, device=device)
            
            b_grid_ids = grid_ids[b]
            b_scores = scores[b]
            
            valid_pts = b_grid_ids >= 0
            if valid_pts.any():
                for i in torch.where(valid_pts)[0]:
                    g_id = b_grid_ids[i]
                    if b_scores[i] > grid_max_scores[g_id]:
                        grid_max_scores[g_id] = b_scores[i]
                        grid_max_idx[g_id] = i
            
            selected_mask = grid_max_idx >= 0
            selected_idx = grid_max_idx[selected_mask]
            
            num_selected = len(selected_idx)
            if num_selected < top_k:
                mask = torch.ones(N, dtype=torch.bool, device=device)
                mask[selected_idx] = False
                
                remaining_scores = b_scores.clone()
                remaining_scores[~mask] = -1e10 
                
                _, extra_idx = torch.topk(remaining_scores, top_k - num_selected)
                batch_final_idx = torch.cat([selected_idx, extra_idx])
            else:
                batch_final_idx = selected_idx[:top_k]
                
            final_indices.append(batch_final_idx)

        indices = torch.stack(final_indices) # [B, 128]
        
        final_feat = torch.gather(desc, 1, indices.unsqueeze(-1).expand(-1, -1, 256))
        final_kpts = torch.gather(kpts, 1, indices.unsqueeze(-1).expand(-1, -1, 2))
        
        return final_feat, final_kpts, indices
        