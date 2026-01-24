import torch
import torch.nn as nn
import torch.nn.functional as F
from lietorch import SE3
from torch_geometric.nn import GATv2Conv

class GeometricGAT(nn.Module):
    def __init__(self, in_channels = 256, hidden_channels = 256, out_channels = 256, heads = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels // heads,
                               heads=heads, edge_dim=3)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, 
                               heads=heads, edge_dim=3, concat=True)
        self.projector = nn.Linear(out_channels * heads, out_channels)
        self.SiLU = nn.SiLU()
        nn.init.constant_(self.projector.bias, 0.1)
    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.SiLU(x)
        x, (edge_index, alpha) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        x = self.projector(x)
        
        return x, alpha
    
class CorrBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fmap1, fmap2):
        d_k = fmap1.shape[-1]
        attn = torch.matmul(fmap1, fmap2.transpose(-1,-2)) / (d_k** 0.5)

        prob = torch.softmax(attn, dim=-1)

        corr_feat = torch.matmul(prob, fmap2)

        return corr_feat


class CyclicErrorModule(nn.Module):
    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline
        self.stereo_offset = None
        self.stereo_inv = None

    def backproject(self, kpts, depth, intrinsics):
        """[B, N, 2], [B, N, 1], [B, 4] -> [B, N, 3]"""
        fx, fy, cx, cy = intrinsics.split(1, dim=-1)
        z = depth
        x = (kpts[..., 0:1] - cx.unsqueeze(1)) * z / fx.unsqueeze(1)
        y = (kpts[..., 1:2] - cy.unsqueeze(1)) * z / fy.unsqueeze(1)
        return torch.cat([x, y, z], dim=-1)

    def project(self, pts_3d, intrinsics):
        """[B, N, 3], [B, 4] -> [B, N, 2]"""
        fx, fy, cx, cy = intrinsics.split(1, dim=-1)
        x, y, z = pts_3d.split(1, dim=-1)
        # 0 나누기 방지 (epsilon)
        z = torch.clamp(z, min=1e-3)
        px = fx.unsqueeze(1) * (x / z) + cx.unsqueeze(1)
        py = fy.unsqueeze(1) * (y / z) + cy.unsqueeze(1)
        return torch.cat([px, py], dim=-1)

    def forward(self, kpts, depth, poses, intrinsics):
        device = kpts.device
        
        # 1. 스테레오 오프셋 초기화
        if self.stereo_offset is None or self.stereo_offset.device != device:
            disp_vec = torch.tensor([self.baseline, 0, 0, 0, 0, 0, 1.0], device=device)
            raw_se3 = SE3.InitFromVec(disp_vec) 
            self.stereo_offset = raw_se3.view((1, 1)) 
            self.stereo_inv = self.stereo_offset.inv()

        # 1. 역투영 (Unprojection)
        pts_3d_Lt = self.backproject(kpts, depth, intrinsics) # [B, N, 3]

        # 2. SE3 객체들의 차원 맞추기 (AssertionError 방지 핵심)
        curr_poses = poses.view((poses.shape[0], 1)) # [B, 1]
        

        # 3. Cycle 연산 수행
        pts_3d_Rt = self.stereo_offset.act(pts_3d_Lt)
        pts_3d_Rt1 = curr_poses.act(pts_3d_Rt)
        pts_3d_Lt1 = self.stereo_inv.act(pts_3d_Rt1)
        pts_3d_Lt_final = curr_poses.inv().act(pts_3d_Lt1)
        kpts_Lt_final = self.project(pts_3d_Lt_final, intrinsics)
        return kpts_Lt_final - kpts
    
class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(770, 512),
            nn.SiLU(),
            nn.Linear(512, hidden_dim),
            nn.SiLU()
        )
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2)
        )
        self.weight_head = SafeWeightHead(hidden_dim)
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() 
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, h, c_temp, c_stereo, e_proj, f_Lt, edges=None, edge_attr=None):
        
        # 기존 c_geo 대신 f_Lt를 사용하여 concat
        x = torch.cat([c_temp, c_stereo, e_proj, f_Lt], dim=-1) # [B, N, D]
        
        if x.dim() == 3:
            B, N, _ = x.shape
        else:
            B = 1
            N, _ = x.shape
            x = x.unsqueeze(0)
            h = h.unsqueeze(0)

        D_h = h.shape[-1]

        # 3. Flatten 연산 (GRU 입력용)
        x_flat = x.view(B * N, -1)
        h_flat = h.view(B * N, -1)

        # 4. Neural Network 연산
        x_flat = self.encoder(x_flat)
        h_flat = self.gru(x_flat, h_flat)

        # 5. 원래 3D 모양으로 복구
        h = h_flat.view(B, N, D_h)

        # 6. Heads 연산
        r = self.residual_head(h)    
        w = self.weight_head(h)        
        a_p = self.alpha_pose_head(h).mean(dim=1) + 1e-3
        a_d = self.alpha_depth_head(h) + 1e-3

        return h, r, w, a_p, a_d
class DBASolver(nn.Module):
    def __init__(self): 
        super().__init__()

    def forward(self, r, w, J_p, J_d, lmbda):
        # w: [B, N, 2], r: [B, N, 2], J_p: [B, N, 2, 6], J_d: [B, N, 2, 1]
        W = w.unsqueeze(-1) # [B, N, 2, 1] (Weighting mask)

        # 1. Hessian & Gradient 블록 계산
        eps_stable = 1e-7
        
        H_pp = torch.matmul(J_p.transpose(-1, -2), W * J_p).sum(dim=1) # [B, 6, 6]
        H_pd = torch.matmul(J_p.transpose(-1, -2), W * J_d)           # [B, N, 6, 1]
        H_dd = torch.matmul(J_d.transpose(-1, -2), W * J_d).squeeze(-1) # [B, N, 1]

        g_p = torch.matmul(J_p.transpose(-1, -2), W * r.unsqueeze(-1)).sum(dim=1)  # [B, 6, 1]
        g_d = torch.matmul(J_d.transpose(-1, -2), W * r.unsqueeze(-1)).squeeze(-1) # [B, N, 1]

        # 2. Levenberg-Marquardt Damping (lmbda) 적용
         
        H_pp = H_pp + lmbda * torch.eye(6, device=H_pp.device).unsqueeze(0)
        H_dd = H_dd + lmbda # H_dd는 [B, N, 1], safe_lmbda는 스칼라이므로 바로 더해짐

        # 3. Schur Complement를 이용한 차원 축소 연산
        # inv_H_dd = 1 / H_dd
        inv_H_dd = 1.0 / (H_dd + 1e-7) # [B, N, 1]
        H_pd_invHdd = H_pd * inv_H_dd.unsqueeze(-1) # [B, N, 6, 1]

        # Reduced Camera Matrix (H_eff) 계산
        H_eff = H_pp - torch.matmul(H_pd_invHdd, H_pd.transpose(-1, -2)).sum(dim=1)
        # 수치적 대칭성 강제 보정
        H_eff = 0.5 * (H_eff + H_eff.transpose(-1, -2))
        
        # g_eff 계산
        g_eff = g_p - (H_pd_invHdd * g_d.unsqueeze(-1)).sum(dim=1)

        # 4. 선형 시스템 풀기 (H_eff * delta_pose = g_eff)
        # H_eff가 여전히 불안정할 수 있으므로 작은 Ridge(eps)를 추가합니다.
        eps_ridge = 1e-3
        diag_idx = torch.arange(H_eff.shape[-1], device=H_eff.device)
        H_eff[:, diag_idx, diag_idx] += eps_ridge
        
        try:
            # linalg.solve가 더 빠르고 안정적입니다.
            delta_pose = torch.linalg.solve(H_eff, g_eff)
        except torch._C._LinAlgError:
            # 행렬이 깨졌을 경우 업데이트를 포기하고 0을 반환 (NaN 확산 방지)
            delta_pose = torch.zeros_like(g_eff)

        # 5. 최종 delta_depth 계산
        # v = H_pd^T * delta_pose
        v = torch.matmul(H_pd.transpose(-1, -2), delta_pose.unsqueeze(1)).squeeze(-1)
        delta_depth = inv_H_dd * (g_d - v)

        return delta_pose.squeeze(-1), delta_depth
    
class PoseDepthUpdater(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d):
        B = curr_pose.shape[0]
        new_depth = curr_depth + a_d * delta_depth
        new_depth = torch.clamp(new_depth, min=0.1)

        scaled_delta = a_p * delta_pose

        delta_SE3 = SE3.exp(scaled_delta)
        new_pose = delta_SE3 * curr_pose

        return new_pose, new_depth
    

class GraphUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.spatial_gat = GeometricGAT(in_channels=258, out_channels=hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, 2)
        )
        self.weight_head = SafeWeightHead(hidden_dim)
        
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid()
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, h, e_proj, f_Lt, edges, edge_attr):
        device = f_Lt.device

        x = torch.cat([e_proj, f_Lt], dim=-1) 
        
        B_total, N, D = f_Lt.shape
        x_flat = x.reshape(-1, x.size(-1))
        
        flat_edges_list = []
        flat_attr_list = []
        for i in range(B_total):
            # 1. edges를 GPU로 이동
            e = edges[i] if isinstance(edges[i], torch.Tensor) else torch.tensor(edges[i])
            flat_edges_list.append(e.to(device) + i * N)
            
            # 2. edge_attr를 GPU로 이동
            a = edge_attr[i] if isinstance(edge_attr[i], torch.Tensor) else torch.tensor(edge_attr[i])
            flat_attr_list.append(a.to(device))
            
        # 3. 결합된 텐서들이 확실히 GPU에 있도록 보장
        edges_combined = torch.cat(flat_edges_list, dim=1).to(device)
        edge_attr_combined = torch.cat(flat_attr_list, dim=0).to(device)
        
        x_spatial_flat, _ = self.spatial_gat(x_flat, edges_combined, edge_attr_combined)
        x_spatial = x_spatial_flat.view(B_total, N, -1)
        
        # 3. GRU 연산: "이전 루프의 수정 히스토리를 반영"
        h_flat = self.gru(x_spatial.reshape(-1, x_spatial.size(-1)), 
                        h.reshape(-1, h.size(-1)))
        h_new = h_flat.view(B_total, N, -1)
        
        # 4. Heads: 수정량(r)과 신뢰도(w), 그리고 보폭(alpha) 계산
        r = self.residual_head(h_new) # [B, N, 2] -> dx, dy 수정 제안
        w = self.weight_head(h_new)   # [B, N, 2] -> 각 점의 신뢰도 가중치
        
        # 포즈 업데이트를 위한 대표값 추출 (모든 점의 의견을 평균)
        a_p = self.alpha_pose_head(h_new).mean(dim=1) + 1e-4 # [B, 1]
        a_d = self.alpha_depth_head(h_new) + 1e-4            # [B, N, 1]

        return h_new, r, w, a_p, a_d


class SafeWeightHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, 2), nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x) * 0.95 + 0.05
    

class EpipolarCrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.dim = feature_dim
        # self.heads = heads
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.merge = nn.Linear(feature_dim, feature_dim)

    def forward(self, nodes_L, nodes_R, kpts_L, kpts_R):
        """
        nodes_L: [N, C] (Left 이미지 노드 특징)
        nodes_R: [M, C] (Right 이미지 노드 특징)
        kpts_L: [N, 2] (u, v 좌표)
        kpts_R: [M, 2] (u, v 좌표)
        """

        N = nodes_L.shape[0]

        Q = self.q_proj(nodes_L)  # [N, C]
        K = self.k_proj(nodes_R)  # [M, C]
        V = self.v_proj(nodes_R)  # [M, C]

        # --- 에피폴라 제약 설정 ---
        dist_v = torch.abs(kpts_L[:, 1:2] - kpts_R[:, 1].unsqueeze(0)) # Y축 차이
        dist_u = kpts_L[:, 0:1] - kpts_R[:, 0].unsqueeze(0)            # X축 차이 (Disparity)
        mask = (dist_v < 3.0) & (dist_u > 0) & (dist_u < 192)

        attn = torch.matmul(Q, K.transpose(0,1)) / (self.dim ** 0.5)   # [N, M]
        attn = attn.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn, dim=-1)                          # [N, M]
        matched_features = torch.matmul(attn_weights, V)                # [N, C]
        # Soft-Argmax 스타일로 초기 Disparity(시차) 계산
        # u_L - u_R 값들을 가중평균
        init_disparity = torch.sum(attn_weights * dist_u, dim=-1, keepdim=True) # [N, 1]
        confidence = mask.any(dim=-1, keepdim=True).float()

        return self.merge(matched_features), init_disparity, confidence
    
class StereoDepthModule(nn.Module):
    def __init__(self,  focal_length, baseline):
        super().__init__()
        self.fB = focal_length * baseline

    def forward(self, nodes_L, matched_features, init_disp, confidence):
        '''
        nodes_L : [N, C] 원본 특징
        matched_features : [N, C] 어텐션으로 가져온 스테레오 특징
        init_disp : [N, 1] 계산된 시차
        confidence : [N, 1] 매칭 신뢰도
        '''
        depth = self.fB / (init_disp + 1e-6)
        inv_depth = 1.0 / (depth + 1e-6)
        # 원본 정보 + 스테레오 문맥 정보 + 기하학적 정보(깊이, 신뢰도)
        updated_nodes = torch.cat([nodes_L, matched_features, inv_depth, confidence], dim=-1) # [N, C + C + 2]

        return updated_nodes, depth
    
class TemporalCrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.dim = feature_dim
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.merge = nn.Linear(feature_dim, feature_dim)

    def forward(self, nodes_t, nodes_t1, kpts_t1_pred, kpts_t1_actual, iter_idx):
        """
        nodes_t: [N, C] (L_t 노드 특징)
        nodes_t1: [M, C] (L_t1 노드 특징 후보군)
        kpts_t1_pred: [N, 2] (현재 포즈 가설로 투영된 예상 위치)
        kpts_t1_actual: [M, 2] (L_t1에서 실제 추출된 특징점 좌표)
        iter_idx: int (현재 루프 횟수, 0이면 루프 진입 전)
        """
        N = nodes_t.shape[0]
        if iter_idx == 0:
            win_size = 64.0
        else:
            win_size = max(4.0, 32.0 / (2 ** (iter_idx - 1)))
        
        Q = self.q_proj(nodes_t)   # [N, C]
        K = self.k_proj(nodes_t1)  # [M, C]
        V = self.v_proj(nodes_t1)  # [M, C]

        dist_u = torch.abs(kpts_t1_pred[:, 0:1] - kpts_t1_actual[:, 0].unsqueeze(0))
        dist_v = torch.abs(kpts_t1_pred[:, 1:2] - kpts_t1_actual[0, 1].unsqueeze(0))

        mask = (dist_u < win_size) & (dist_v < win_size)

        attn = torch.matmul(Q, K.transpose(0, 1)) / (self.dim ** 0.5)
        attn = attn.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn, dim=-1)      # [N, M]

        matched_features = torch.matmul(attn_weights, V)    #[N, C]
        matched_kpts = torch.matmul(attn_weights, kpts_t1_actual) # [N, 2]

        flow_residual = matched_kpts - kpts_t1_pred

        confidence = mask.any(dim=-1, keepdim=True).float()

        return self.merge(matched_features), flow_residual, confidence