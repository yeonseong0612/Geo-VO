import torch
import torch.nn as nn
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
            
            # [수정] unsqueeze 대신 view를 사용하되, 인자를 반드시 '튜플'로 전달
            # 이렇게 하면 lietorch 내부의 '리스트 + 튜플' 에러를 피할 수 있습니다.
            self.stereo_offset = raw_se3.view((1, 1)) 
            self.stereo_inv = self.stereo_offset.inv()

        # 1. 역투영 (Unprojection)
        pts_3d_Lt = self.backproject(kpts, depth, intrinsics) # [B, N, 3]

        # 2. SE3 객체들의 차원 맞추기 (AssertionError 방지 핵심)
        # poses가 [B] 형태라면 view나 unsqueeze로 [B, 1]로 만듭니다.
        # lietorch SE3 객체는 .view()를 사용합니다.
        curr_poses = poses.view((poses.shape[0], 1)) # [B, 1]
        
        # stereo_offset은 이미 위에서 [1, 1]로 만드셨을 겁니다.
        # 만약 에러가 난다면 이 녀석도 체크가 필요합니다.

        # 3. Cycle 연산 수행
        # [1, 1] act [B, N, 3] -> [B, N, 3]
        pts_3d_Rt = self.stereo_offset.act(pts_3d_Lt)
        
        # [B, 1] act [B, N, 3] -> [B, N, 3] (이제 여기서 에러가 안 납니다!)
        pts_3d_Rt1 = curr_poses.act(pts_3d_Rt)
        
        # Lt1 방향으로 돌아오는 연산들도 동일하게 적용
        pts_3d_Lt1 = self.stereo_inv.act(pts_3d_Rt1)
        pts_3d_Lt_final = curr_poses.inv().act(pts_3d_Lt1)

        # 4. 재투영 (Projection)
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
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
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
        a_p = self.alpha_pose_head(h).mean(dim=1) 
        a_d = self.alpha_depth_head(h) 

        return h, r, w, a_p, a_d
class DBASolver(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, r, w, J_p, J_d, lmbda):
        # w: [B, N, 2], r: [B, N, 2], J_p: [B, N, 2, 6], J_d: [B, N, 2, 1]
        W = w.unsqueeze(-1) # [B, N, 2, 1] (Weighting mask)

        # 1. Hessian & Gradient 블록 계산
        # 안정성을 위해 연산 전 단계에서 아주 작은 값(eps)을 고려합니다.
        eps_stable = 1e-7
        
        H_pp = torch.matmul(J_p.transpose(-1, -2), W * J_p).sum(dim=1) # [B, 6, 6]
        H_pd = torch.matmul(J_p.transpose(-1, -2), W * J_d)           # [B, N, 6, 1]
        H_dd = torch.matmul(J_d.transpose(-1, -2), W * J_d).squeeze(-1) # [B, N, 1]

        g_p = torch.matmul(J_p.transpose(-1, -2), W * r.unsqueeze(-1)).sum(dim=1)  # [B, 6, 1]
        g_d = torch.matmul(J_d.transpose(-1, -2), W * r.unsqueeze(-1)).squeeze(-1) # [B, N, 1]

        # 2. Levenberg-Marquardt Damping (lmbda) 적용
        # H_dd는 대각 성분이므로 lmbda를 더해 역수 연산 시 안정성을 확보합니다.
        H_pp = H_pp + lmbda.view(-1, 1, 1) * torch.eye(6, device=H_pp.device)
        H_dd = H_dd + lmbda.view(-1, 1, 1) + eps_stable

        # 3. Schur Complement를 이용한 차원 축소 연산
        # inv_H_dd = 1 / H_dd
        inv_H_dd = 1.0 / H_dd # [B, N, 1]
        H_pd_invHdd = H_pd * inv_H_dd.unsqueeze(-1) # [B, N, 6, 1]

        # Reduced Camera Matrix (H_eff) 계산
        H_eff = H_pp - torch.matmul(H_pd_invHdd, H_pd.transpose(-1, -2)).sum(dim=1)
        # 수치적 대칭성 강제 보정
        H_eff = 0.5 * (H_eff + H_eff.transpose(-1, -2))
        
        # g_eff 계산
        g_eff = g_p - (H_pd_invHdd * g_d.unsqueeze(-1)).sum(dim=1)

        # 4. 선형 시스템 풀기 (H_eff * delta_pose = g_eff)
        # H_eff가 여전히 불안정할 수 있으므로 작은 Ridge(eps)를 추가합니다.
        eps_ridge = 1e-4
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
        self.spatial_gat = GeometricGAT(in_channels=770, out_channels=hidden_dim)
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, 2)
        )
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, 2), nn.Sigmoid()
        )
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Softplus()
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(), nn.Linear(128, 1), nn.Softplus()
        )

    def forward(self, h, c_temp, c_stereo, e_proj, f_Lt, edges, edge_attr):
        # f_Lt: [Total_N, D] (이제 2차원으로 들어옴)
        
        # 1. 모든 특징 결합
        x = torch.cat([c_temp, c_stereo, e_proj, f_Lt], dim=-1) # [Total_N, 770]
        if x.dim() == 3:
            x = x.view(-1, x.shape[-1])
        
        # 2. GAT 연산 (x_flat 과정 필요 없음)
        x_spatial, _ = self.spatial_gat(x, edges, edge_attr)
        
        # 3. GRU 업데이트 (h도 [Total_N, D] 여야 함)
        h_new = self.gru(x_spatial, h)
        
        # 4. Heads 연산
        r = self.residual_head(h_new)
        w = self.weight_head(h_new)
        
        a_p = self.alpha_pose_head(h_new) 
        a_d = self.alpha_depth_head(h_new) 

        return h_new, r, w, a_p, a_d