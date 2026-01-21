import torch
import torch.nn as nn
from lietorch import SE3
from torch_geometric.nn import GATv2Conv

class GeometricGAT(nn.Module):
    def __init__(self, in_channels = 256, hidden_channels = 256, out_channels = 256, heads = 4):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels, hidden_channels // heads,
                               heads=heads, edge_dim=1)
        self.conv2 = GATv2Conv(hidden_channels, out_channels, 
                               heads=heads, edge_dim=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x, (edge_index, alpha) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        
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
    def __init__(self, baseline, device):
        super().__init__()
        # 1. 고정된 스테레오 변환을 미리 GPU에 생성 (딱 한 번만)
        disp_vec = torch.tensor([baseline, 0, 0, 0, 0, 0, 1.0], device=device)
        self.stereo_offset = SE3.InitFromVec(disp_vec).unsqueeze(0) # [1, 1]
        self.stereo_inv = self.stereo_offset.inv()
        
    def backproject(self, kpts, depth, intrinsics):
        # [B, 1, 1] 형태로 차원 정리
        fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3]
        fx = fx.view(-1, 1, 1)
        fy = fy.view(-1, 1, 1)
        cx = cx.view(-1, 1, 1)
        cy = cy.view(-1, 1, 1)

        u, v = kpts[..., 0:1], kpts[..., 1:2]
        
        # depth가 [B, N]으로 들어올 경우를 대비해 안전하게 unsqueeze
        if depth.dim() == 2:
            depth = depth.unsqueeze(-1)

        X = (u - cx) * depth / fx
        Y = (v - cy) * depth / fy
        Z = depth
        return torch.cat([X, Y, Z], dim=-1)

    def project(self, pts_3d, intrinsics):
        fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3]
        fx = fx.view(-1, 1, 1)
        fy = fy.view(-1, 1, 1)
        cx = cx.view(-1, 1, 1)
        cy = cy.view(-1, 1, 1)

        X, Y, Z = pts_3d[..., 0:1], pts_3d[..., 1:2], pts_3d[..., 2:3]
        Z = Z + 1e-8

        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy
        return torch.cat([u, v], dim=-1)

    def forward(self, kpts_Lt, depth_Lt, rel_pose, intrinsics):
        # 2. rel_pose가 [B]라면 [B, 1]로 확장하여 N개의 점에 대해 브로드캐스팅 준비
        if len(rel_pose.data.shape) == 2:
            rel_pose = rel_pose[:, None]

        # 3. 모든 연산은 GPU 커널에서 텐서 단위로 병렬 처리됨
        pts_3d_Lt = self.backproject(kpts_Lt, depth_Lt, intrinsics) # [B, N, 3]

        # Cycle: Lt -> Rt -> Rt1 -> Lt1 -> Lt_final
        # 아래 act() 연산들은 lietorch의 최적화된 CUDA 커널을 사용합니다.
        pts_3d_Rt = self.stereo_offset.act(pts_3d_Lt)
        pts_3d_Rt1 = rel_pose.act(pts_3d_Rt)
        pts_3d_Lt1 = self.stereo_inv.act(pts_3d_Rt1)
        pts_3d_Lt_final = rel_pose.inv().act(pts_3d_Lt1)

        kpts_Lt_final = self.project(pts_3d_Lt_final, intrinsics)
        return kpts_Lt_final - kpts_Lt
    
class UpdateBlock(nn.Module):
    def __init__(self, hidden_dim = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(770, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
            nn.Sigmoid()
        )
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus() 
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )

    def forward(self, h, c_temp, c_stereo, e_proj, c_geo):
        x = torch.cat([c_temp, c_stereo, e_proj, c_geo], dim=-1) # [B, N, 769]
        if x.dim() == 3:
            B, N, _ = x.shape
        else:
            # 만약 2차원 [N, 769]로 들어왔다면 B=1로 간주
            B = 1
            N, _ = x.shape
            # 연산을 위해 3차원으로 강제 확장 [1, N, D]
            x = x.unsqueeze(0)
            h = h.unsqueeze(0)

        D_h = h.shape[-1]

        # 3. Flatten 연산 (GRU는 2D를 원함)
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
        W = w.unsqueeze(-1) # [B, N, 2, 1]

        # 1. Hessian & Gradient 블록
        H_pp = torch.matmul(J_p.transpose(-1, -2), W * J_p).sum(dim=1) # [B, 6, 6]
        H_pd = torch.matmul(J_p.transpose(-1, -2), W * J_d)           # [B, N, 6, 1]
        H_dd = torch.matmul(J_d.transpose(-1, -2), W * J_d).squeeze(-1) # [B, N, 1]

        g_p = torch.matmul(J_p.transpose(-1, -2), W * r.unsqueeze(-1)).sum(dim=1) # [B, 6, 1]
        g_d = torch.matmul(J_d.transpose(-1, -2), W * r.unsqueeze(-1)).squeeze(-1) # [B, N, 1]

        B = H_pp.shape[0]
        H_pp = H_pp + lmbda * torch.eye(6, device=H_pp.device).unsqueeze(0)
        H_dd = H_dd + lmbda

        inv_H_dd = 1.0 / torch.clamp(H_dd, min=1e-6) # [B, N, 1]
        H_pd_invHdd = H_pd * inv_H_dd.unsqueeze(-1) # [B, N, 6, 1]

        H_eff = H_pp - torch.matmul(H_pd_invHdd, H_pd.transpose(-1, -2)).sum(dim=1)
        H_eff = 0.5 * (H_eff + H_eff.transpose(-1, -2))
        g_eff = g_p - (H_pd_invHdd * g_d.unsqueeze(-1)).sum(dim=1)

        eps = 1e-4
        identity = torch.eye(H_eff.shape[-1], device=H_eff.device).expand_as(H_eff)
        H_eff_stable = H_eff + eps * identity
        try:
            delta_pose = torch.linalg.solve(H_eff_stable, g_eff)
        except torch._C._LinAlgError:
            delta_pose = torch.zeros_like(g_eff)
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
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 2)
        )
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 2), nn.Sigmoid()
        )
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Softplus()
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 1), nn.Softplus()
        )

    def forward(self, h, c_temp, c_stereo, e_proj, f_Lt, edges, edge_attr):
        B, N, _ = f_Lt.shape
        

        x = torch.cat([c_temp, c_stereo, e_proj, f_Lt], dim=-1) 
        
        x_flat = x.view(-1, x.shape[-1])
        x_spatial, _ = self.spatial_gat(x_flat, edges, edge_attr)
        
        h_flat = h.view(-1, h.shape[-1])
        h_new_flat = self.gru(x_spatial, h_flat)
        h_new = h_new_flat.view(B, N, -1)
        
        r = self.residual_head(h_new)
        w = self.weight_head(h_new)
        
        a_p = self.alpha_pose_head(h_new).mean(dim=1) 
        a_d = self.alpha_depth_head(h_new) 

        return h_new, r, w, a_p, a_d