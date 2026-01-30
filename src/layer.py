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
        # 1. 입력받은 in_channels가 256인지 320인지 명확히 해야 합니다.
        # 만약 PoseInitializer에서 320을 넣어준다면, 여기서 64를 더하면 안 됩니다.
        
        # [추천] 차원을 명시적으로 계산
        self.node_base_dim = in_channels # PoseInitializer에서 256을 넣어준다고 가정
        self.pos_dim = 64
        self.total_in_dim = self.node_base_dim + self.pos_dim # 256 + 64 = 320

        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, 32),
            nn.LayerNorm(32), 
            nn.SiLU(),
            nn.Linear(32, self.pos_dim),
            nn.LayerNorm(self.pos_dim)
        )

        # GATv2Conv의 입력은 반드시 x_combined의 차원인 320이어야 합니다.
        self.conv = GATv2Conv(
            self.total_in_dim, # <--- 여기가 384로 되어있을 가능성이 큽니다. 320으로 고정!
            hidden_dim // heads,
            heads=heads, 
            edge_dim=3,
            add_self_loops=False
        )
        self.res_proj = nn.Linear(self.total_in_dim, hidden_dim)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.post_norm = nn.LayerNorm(hidden_dim)
        self.projector = nn.Linear(hidden_dim, 256)
        self.SiLU = nn.SiLU()

    def forward(self, x, edge_index, kpts, pts_3d, edge_attr=None):
        device = x.device
        
        # [수정 1] 입력 데이터 자체에 아주 미세한 노이즈 추가 (LayerNorm/Norm 보호)
        # 0인 데이터가 0인 상태로 연산에 들어가는 것을 원천 차단
        kpts = kpts + torch.randn_like(kpts) * 1e-7
        pts_3d = pts_3d + torch.randn_like(pts_3d) * 1e-7
        
        # [수정 2] 좌표 정규화 및 깊이 제한
        norm_uv = kpts / torch.tensor([1216.0, 352.0], device=device)
        depth = torch.clamp(pts_3d[:, 2:3], min=0.1, max=100.0)
        
        # [Step 1] 위치 특징 추출 (LayerNorm 폭주 방지용 epsilon)
        pos_input = torch.cat([norm_uv, depth], dim=-1)
        pos_feat = self.pos_encoder(pos_input)
        
        # [N, 320] 결합
        x_combined = torch.cat([x, pos_feat], dim=-1)

        # [Step 2] Edge_info 계산 (Gradient NaN 보호)
        if edge_attr is None and edge_index is not None:
            src, dst = edge_index[0], edge_index[1]
            rel_uv = norm_uv[dst] - norm_uv[src]
            # 안전한 norm 계산
            dist = torch.sqrt(torch.sum(rel_uv**2, dim=-1, keepdim=True) + 1e-9)
            edge_attr = torch.cat([rel_uv, dist], dim=-1)

        # [Step 4] GAT 연산 - NaN 전파 방지
        if torch.isnan(x_combined).any():
            x_combined = torch.where(torch.isnan(x_combined), torch.zeros_like(x_combined), x_combined)

        out, _ = self.conv(x_combined, edge_index, edge_attr, return_attention_weights=True)
        
        # [Step 5] Residual Connection & Final Proj
        out = self.norm(out)
        identity = self.res_proj(x_combined)
        
        # 합산 직전 체크
        out = self.post_norm(self.SiLU(out + identity)) 
        out = self.projector(out)
        
        return out, None, edge_attr

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
        # 1. node_feat 차원 강제 보정 [B, N, C]
        if node_feat.dim() == 2: # [N, C]인 경우 배치 차원 추가
            node_feat = node_feat.unsqueeze(0)
            
        B = node_feat.shape[0]
        all_weights = []
        all_normals = []

        for b in range(B):
            # [수정] tri_indices[b]가 텐서인지 리스트인지에 따라 안전하게 처리
            tris_full = tri_indices[b] # [max_T, 3]
            mask = tris_full[:, 0] > -1 # -1이 아닌 유효한 행만 선택
            tris = tris_full[mask]
            if isinstance(tris, list):
                tris = torch.tensor(tris, device=node_feat.device)
            
            # 삼각형이 없는 경우 예외 처리
            if tris.shape[0] == 0:
                all_weights.append(torch.zeros((0, 1), device=node_feat.device))
                all_normals.append(torch.zeros((0, 3), device=node_feat.device))
                continue

            # [핵심] 배치 인덱싱을 명확하게 수행
            # node_feat[b] -> [N, 256]
            f1 = node_feat[b, tris[:, 0]] # [T, 256]
            f2 = node_feat[b, tris[:, 1]] # [T, 256]
            f3 = node_feat[b, tris[:, 2]] # [T, 256]

            # 2. Concat Feature [T, 768]
            f_tri = torch.cat([f1, f2, f3], dim=-1)

            # 3. MLP 수행
            feat = self.mlp(f_tri)

            # 4. Result 계산
            weights = self.weight_head(feat)    # [T, 1]
            normals = self.normal_head(feat)    # [T, 3]
            # [추가] 정규화 시 0으로 나누기 방지
            norm_val = torch.norm(normals, p=2, dim=-1, keepdim=True)
            normals = torch.where(norm_val > 1e-8, normals / norm_val, torch.zeros_like(normals))

            all_weights.append(weights)
            all_normals.append(normals)
            
        return all_weights, all_normals
        
class PoseInitializer(nn.Module):
    def __init__(self, in_channels=256, node_dim=256):
        super().__init__()
        self.gat = GeometricGAT(in_channels=in_channels, hidden_dim=256)
        self.tri_head = TriangleHead(node_dim=node_dim)

    def forward(self, descs, kpts, pts_3d, tri_indices, kpts_tp1, intrinsics):
        B, N, _ = descs.shape
        device = descs.device

        # [수정 1] 배치 전체의 유효한 삼각형만 모아서 에지를 만듭니다.
        # tri_indices가 [B, max_T, 3] 텐서이므로, 마스크를 통해 리스트로 분리합니다.
        valid_tris_list = []
        for b in range(B):
            tris_full = tri_indices[b]
            mask = tris_full[:, 0] > -1
            valid_tris_list.append(tris_full[mask])

        # [Step 1] GAT : 에지 생성 및 특징 업데이트
        # tri_indices_to_edges 내부에서 배치 오프셋 처리가 되어있어야 합니다.
        edges = tri_indices_to_edges(valid_tris_list, B, N, device)
        
        node_feat_flat, _, edge_attr = self.gat(
            x=descs.view(-1, 256), edge_index=edges,
            kpts=kpts.view(-1, 2), pts_3d=pts_3d.view(-1, 3)
        )
        node_feat = node_feat_flat.view(B, N, 256)
        
        # [Step 2] Triangle Weights (유효한 삼각형 리스트 전달)
        weights_list, _ = self.tri_head(node_feat, valid_tris_list)

        final_R_list, final_tri_weights, final_vp_conf = [], [], []

        # [Step 3] 배치별 초기 Pose 결정 루프
        for b in range(B):
            tris = valid_tris_list[b].to(device) # 이미 마스킹된 삼각형 사용
            raw_w = weights_list[b]
            
            # [Step 2] 정규화 좌표 계산 (미분 불필요)
            fx, fy, cx, cy = intrinsics[b]
            p_norm = torch.stack([
                (kpts_tp1[b, :, 0] - cx) / (fx + 1e-8),
                (kpts_tp1[b, :, 1] - cy) / (fy + 1e-8),
                torch.ones(N, device=device)
            ], dim=-1)

            # [Step 3] 소실점 투표 (수치 안정성을 위해 미분 끊기)
            # w_static: 투표 로직이 GAT를 직접 흔드는 것을 방지
            w_static = raw_w.detach() 
            K_j = compute_individual_Kj(tris, pts_3d[b], p_norm) + torch.randn(tris.shape[0], 3, 3, device=device)*1e-5
            if torch.isnan(K_j).any(): print(f"!!! Batch {b}: K_j has NaN")
            R_cands = batch_svd(K_j)
            if torch.isnan(R_cands).any(): 
                print(f"🚨 Batch {b}: batch_svd output has NaN! Check compute_individual_Kj.")
            
            # xv_j: 나눗셈 폭주 방지
            xv_j = fx * (R_cands[:, 0, 2] / torch.clamp(R_cands[:, 2, 2], min=0.01)) + cx
            xv_j = torch.clamp(xv_j, -2000, 4000)
            
            # xv_star: 최적 소실점 (상수로 취급)
            xv_star = differentiable_voting(xv_j, w_static, sigma=2.0).detach() 

            # s_j: 투표 신뢰도 (미분 차단)
            dist_sq = torch.clamp((xv_j - xv_star)**2, max=100.0)
            s_j = torch.exp(-dist_sq / (2 * 2.0**2)).unsqueeze(-1).detach()

            # [Step 4] 최종 R_init 계산 (GAT로 미분이 흐르는 핵심 구간)
            # s_j는 상수로 취급하여 GAT가 '어떤 삼각형이 투표를 잘했는지'에 집중하게 함
            combined_w = raw_w * s_j 
            R_init = estimate_rotation_svd_differentiable(combined_w, tris, pts_3d[b], p_norm)
            if torch.isnan(R_init).any():
                print(f"🚨 Batch {b}: R_init is NaN! SVD gradient might be exploding.")

            # [Step 5] 결과 정리
            v_conf = torch.tanh(torch.zeros((N, 1), device=device).scatter_add_(
                0, tris.view(-1, 1).expand(-1, 1), s_j.repeat_interleave(3, dim=0)
            ))

            final_R_list.append(R_init)
            final_tri_weights.append(combined_w)
            final_vp_conf.append(v_conf)

        return torch.stack(final_R_list), final_tri_weights, torch.stack(final_vp_conf), edges, edge_attr
    
class DBASolver(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, r, w, J_p, J_d, lmbda, iter_idx):
        """
        iter_idx: 현재 최적화 루프의 인덱스 (0부터 시작)
        """
        B, N, _ = r.shape
        device = r.device
        
        # 1. lmbda 방어
        safe_lmbda = torch.where(torch.isnan(lmbda), torch.tensor(1e2, device=device), lmbda)
        safe_lmbda = torch.clamp(safe_lmbda, min=1e-3)

        # 2. 가중치 처리
        conf = w[..., 0:1].unsqueeze(-1)    
        node_lambda = w[..., 1:2] if w.shape[-1] >= 2 else torch.zeros((B, N, 1), device=device)

        # 3. Hessian 및 Gradient 계산
        H_pp = torch.matmul(J_p.transpose(-1, -2), conf * J_p).sum(dim=1)
        H_pd = torch.matmul(J_p.transpose(-1, -2), conf * J_d)
        H_dd = torch.matmul(J_d.transpose(-1, -2), conf * J_d).squeeze(-1)

        g_p = torch.matmul(J_p.transpose(-1, -2), conf * r.unsqueeze(-1)).sum(dim=1) 
        g_d = torch.matmul(J_d.transpose(-1, -2), conf * r.unsqueeze(-1)).squeeze(-1) 

        # 4. Levenberg-Marquardt Damping
        diag_mask = torch.eye(6, device=device).unsqueeze(0)
        H_pp = H_pp + (safe_lmbda * diag_mask) 
        
        H_dd_safe = torch.clamp(H_dd + safe_lmbda + node_lambda + 1e-4, min=1e-4)
        inv_H_dd = 1.0 / H_dd_safe
        
        H_pd_invHdd = H_pd * inv_H_dd.view(B, N, 1, 1)
        term_to_sub = torch.matmul(H_pd_invHdd, H_pd.transpose(-1, -2)).sum(dim=1)
    
        # 5. Reduced Camera System (Schur Complement)
        H_eff = H_pp - term_to_sub
        g_eff = g_p - (H_pd_invHdd * g_d.unsqueeze(-1)).sum(dim=1)

        # [전략 수정] Warm-up Rotation Refinement
        # iter_idx가 2(혹은 1) 이상일 때만 회전 업데이트를 차단합니다.
        # 초반 0, 1회차 루프에서는 R도 함께 최적화하여 T와의 정렬을 맞춥니다.
        if iter_idx >= 2:
            # 회전(3,4,5)에 대한 Gradient를 0으로 설정
            g_eff[:, 3:] = 0.0
            # 회전 대각 성분에 큰 값을 더해 업데이트 억제
            H_eff[:, 3:, 3:] += 1e8 

        # 6. Ridge Damping 강화
        eps_ridge = 1.0
        H_eff = H_eff + eps_ridge * torch.eye(6, device=device).unsqueeze(0)
        H_eff = H_eff + torch.diag_embed(torch.diagonal(H_eff, dim1=-2, dim2=-1) * 0.01)

        # 7. Linear System Solve
        try:
            delta_pose = torch.linalg.solve(H_eff, g_eff) 
        except RuntimeError:
            delta_pose = torch.zeros_like(g_eff)

        # 8. 최종 delta_depth 계산
        v = torch.matmul(H_pd.transpose(-1, -2), delta_pose.unsqueeze(1)).squeeze(-1)
        delta_depth = inv_H_dd * (g_d - v)
        
        delta_pose = delta_pose.squeeze(-1)
        
        # [방어] 업데이트 값 제한
        delta_pose = torch.clamp(delta_pose, min=-2.0, max=2.0)
        delta_depth = torch.clamp(delta_depth, min=-5.0, max=5.0)

        if torch.isnan(delta_pose).any(): delta_pose = torch.zeros_like(delta_pose)
        if torch.isnan(delta_depth).any(): delta_depth = torch.zeros_like(delta_depth)

        return delta_pose, delta_depth
    
class PoseDepthUpdater(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=100.0):
        super().__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
    
    def forward(self, curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d, iter_idx):
        # 1. Depth Update
        safe_delta_d = torch.clamp(delta_depth, min=-5.0, max=5.0) 
        new_depth = torch.clamp(curr_depth + a_d * safe_delta_d, min=self.min_depth, max=self.max_depth)

        # 2. Pose Update (Conditional Rotation)
        scaled_delta = a_p * torch.tanh(delta_pose / 2.0) * 2.0
        
        # [전략 수정] 초반 2회까지만 R 업데이트 허용, 이후는 T만 업데이트
        if iter_idx < 2:
            # R과 T 모두 업데이트 (기하학적 정렬)
            pure_delta = scaled_delta 
        else:
            # T만 업데이트 (스케일 정밀 보정)
            pure_delta = torch.cat([
                scaled_delta[..., :3], 
                torch.zeros_like(scaled_delta[..., 3:])
            ], dim=-1)

        delta_SE3 = SE3.exp(pure_delta)
        new_pose = curr_pose * delta_SE3 
        return new_pose, new_depth  

class GraphUpdateBlock(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Node(256) + r(2) + tri_w(1) + vp_s(1) = 260
        self.spatial_gat = GeometricGAT(in_channels=260, hidden_dim=hidden_dim)
        self.norm_gat = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 4) # conf, a_p, a_d, (spare)
        )

    def forward(self, h, node_feat, r, tri_w, vp_s, edges, edge_attr, intrinsics, kpts, pts_3d):
        B, N, _ = node_feat.shape
        
        # r_norm 및 입력 결합
        r_norm = r / (intrinsics[:, :2].unsqueeze(1) + 1e-8) 
        x_fused = torch.cat([node_feat, r_norm, tri_w, vp_s], dim=-1)
        x_fused = torch.where(torch.isnan(x_fused), torch.zeros_like(x_fused), x_fused)

        # Spatial GAT
        x_spatial_flat, _, _ = self.spatial_gat(
            x=x_fused.reshape(-1, 260), 
            edge_index=edges, edge_attr=edge_attr,
            kpts=kpts.reshape(-1, 2), pts_3d=pts_3d.reshape(-1, 3)
        )
        
        x_spatial = self.norm_gat(x_spatial_flat.view(B, N, -1))
        h_new_flat = self.gru(
            torch.clamp(x_spatial, -10.0, 10.0).reshape(-1, self.hidden_dim), 
            torch.clamp(h, -10.0, 10.0).reshape(-1, self.hidden_dim)
        )
        # Temporal GRU
        h_new_flat = self.gru(x_spatial.reshape(-1, self.hidden_dim), h.reshape(-1, self.hidden_dim))
        h_new = self.norm_h(torch.clamp(h_new_flat, -50.0, 50.0).view(B, N, -1))

        # Decision
        out = self.head(h_new)
        conf = torch.sigmoid(out[..., 0:1])
        a_p = torch.sigmoid(out[..., 1:2]).mean(dim=1) * 0.1
        a_d = torch.sigmoid(out[..., 2:3]) * 0.1
        
        return h_new, conf, a_p, a_d



