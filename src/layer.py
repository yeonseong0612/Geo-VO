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
    def __init__(self, baseline):
        super().__init__()
        # 정면 기준 오른쪽 카메라로의 이동 (Baseline만큼 X축 이동)
        # KITTI는 보통 왼쪽(Cam2)이 기준이므로 오른쪽(Cam3)은 X축으로 +Baseline 이동
        disp_vec = torch.tensor([baseline, 0, 0, 0, 0, 0, 1.0], dtype=torch.float32)
        self.register_buffer('stereo_offset_vec', disp_vec.unsqueeze(0))

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
        B = kpts_Lt.shape[0]
        
        # 1. 스테레오 오프셋 생성 및 차원 확장 [B, 1, 7]
        # .unsqueeze(1) 대신 [:, None] 또는 [..., None]을 사용합니다.
        stereo_vec = self.stereo_offset_vec.repeat(B, 1)
        stereo_offset = SE3.InitFromVec(stereo_vec)[:, None] 
        
        # 2. rel_pose 차원 확장 [B, 1, 7]
        # SE3 객체에 직접 인덱싱을 하면 내부 data 텐서의 차원이 확장된 새 객체가 반환됩니다.
        if len(rel_pose.data.shape) == 2:
            rel_pose = rel_pose[:, None]

        # 3. Backproject (pts_3d_Lt: [B, N, 3])
        pts_3d_Lt = self.backproject(kpts_Lt, depth_Lt, intrinsics)

        # 4. Cycle Transformation
        # 이제 모두 3차원 텐서 구조(B, N, 3)와 (B, 1, 7)로 맞춰졌습니다.
        pts_3d_Rt = stereo_offset.act(pts_3d_Lt)
        pts_3d_Rt1 = rel_pose.act(pts_3d_Rt)
        pts_3d_Lt1 = stereo_offset.inv().act(pts_3d_Rt1)
        pts_3d_Lt_final = rel_pose.inv().act(pts_3d_Lt1)

        # 5. Project & Error
        kpts_Lt_final = self.project(pts_3d_Lt_final, intrinsics)
        cycle_error_2d = kpts_Lt_final - kpts_Lt 

        return cycle_error_2d
    
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

        # self.pose_gate = nn.Sequential(
        #     nn.Linear(hidden_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128,6)
        # )
        # self.depth_gate = nn.Sequential(
        #     nn.Linear(hidden_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128,1)
        # )
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
        # 디버깅용: 각 입력의 차원을 출력해봅니다 (나중에 삭제)
        # print(f"c_temp: {c_temp.shape}, c_stereo: {c_stereo.shape}, e_proj: {e_proj.shape}, c_geo: {c_geo.shape}")

        # 1. 모든 입력의 마지막 차원을 기준으로 합칩니다.
        # e_proj가 [B, N, 1]인지 꼭 확인해야 합니다.
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

        # 2. Damping
        B = H_pp.shape[0]
        H_pp = H_pp + lmbda * torch.eye(6, device=H_pp.device).unsqueeze(0)
        H_dd = H_dd + lmbda

        # 3. Schur Complement (RCS 구성)
        inv_H_dd = 1.0 / (H_dd + 1e-8) # [B, N, 1]
        H_pd_invHdd = H_pd * inv_H_dd.unsqueeze(-1) # [B, N, 6, 1]

        # H_eff 계산: matmul( [B, N, 6, 1], [B, N, 1, 6] ) -> [B, N, 6, 6]
        H_eff = H_pp - torch.matmul(H_pd_invHdd, H_pd.transpose(-1, -2)).sum(dim=1)
        
        # g_eff 계산: [B, N, 6, 1] * [B, N, 1, 1] -> [B, N, 6, 1]
        g_eff = g_p - (H_pd_invHdd * g_d.unsqueeze(-1)).sum(dim=1)

        # 4. Solve
        eps = 1e-4
        identity = torch.eye(H_eff.shape[-1], device=H_eff.device).expand_as(H_eff)
        H_eff_stable = H_eff + eps * identity
        try:
            delta_pose = torch.linalg.solve(H_eff_stable, g_eff)
        except torch._C._LinAlgError:
            # 만약 그래도 에러가 난다면, 더 큰 댐핑을 주거나 0으로 처리하여 학습 중단을 방지합니다.
            delta_pose = torch.zeros_like(g_eff)
        # 5. Back-substitution (Depth 업데이트 계산)
        # v = H_pd^T * delta_pose -> [B, N, 1, 6] * [B, 1, 6, 1] -> [B, N, 1, 1]
        v = torch.matmul(H_pd.transpose(-1, -2), delta_pose.unsqueeze(1)).squeeze(-1)
        delta_depth = inv_H_dd * (g_d - v)

        return delta_pose.squeeze(-1), delta_depth
    
class PoseDepthUpdater(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d):
        B = curr_pose.shape[0]
        new_depth = curr_depth + a_d * delta_depth

        scaled_delta = a_p * delta_pose

        delta_SE3 = SE3.exp(scaled_delta)
        new_pose = delta_SE3 * curr_pose

        return new_pose, new_depth