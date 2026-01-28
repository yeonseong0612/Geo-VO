import torch
import torch.nn as nn
import torch.nn.functional as F
from lietorch import SE3
from torch_geometric.nn import GATv2Conv

import torch
import torch.nn as nn

class GeometricGAT(nn.Module):
    def __init__(self, desc_dim=256, pos_dim=3, hidden_channels=256, heads=4):
        super().__init__()
        # 1. Positional Encoding (u, v, Z 정보를 고차원으로 투영)
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 64)
        )

        # GAT 입력 차원: Descriptor(256) + Positional(64) = 320
        in_channels = desc_dim + 64
        
        # Residual Projection (320 -> 256)
        self.res_proj = nn.Linear(in_channels, hidden_channels)

        # 2. GAT Layer (기하학적 엣지 속성 edge_dim=3 사용)
        # edge_dim 3: [상대_u, 상대_v, 유클리드_거리]
        self.conv = GATv2Conv(
            in_channels, 
            hidden_channels // heads,
            heads=heads, 
            edge_dim=3,
            add_self_loops=False
        )
        
        self.norm = nn.LayerNorm(hidden_channels)
        self.projector = nn.Linear(hidden_channels, 256) # 최종 R 추정용 피처
        self.SiLU = nn.SiLU()

    def forward(self, x, kpts, pts_3d, edge_index):
        """
        x: [N, 256] (Descriptors)
        kpts: [N, 2] (u, v)
        pts_3d: [N, 3] (X, Y, Z) - 여기서 Z(Depth)만 사용
        edge_index: [2, E]
        """
        # [A] Node Feature 구성: Descs + MLP(u, v, Z)
        # u, v 정규화 (이미지 사이즈 1216x352 기준)
        norm_uv = kpts / torch.tensor([1216.0, 352.0], device=kpts.device)
        depth = pts_3d[:, 2:3] # Depth Z만 추출
        
        pos_input = torch.cat([norm_uv, depth], dim=-1) # [N, 3]
        pos_feat = self.pos_encoder(pos_input) # [N, 64]
        
        node_x = torch.cat([x, pos_feat], dim=-1) # [N, 320]
        
        # [B] Edge Attribute 생성: 상대적 기하 관계 [Δu, Δv, dist]
        src, dst = edge_index[0], edge_index[1]
        rel_uv = norm_uv[dst] - norm_uv[src] # [E, 2]
        dist = torch.norm(rel_uv, dim=-1, keepdim=True) # [E, 1]
        edge_attr = torch.cat([rel_uv, dist], dim=-1) # [E, 3]
        
        # [C] GAT Forward + Residual
        identity = self.res_proj(node_x)
        
        # GAT 통과 (1회)
        out, (edge_index_out, alpha) = self.conv(node_x, edge_index, edge_attr, return_attention_weights=True)
        
        out = self.norm(out)
        out = self.SiLU(out)
        out = out + identity # Residual Connection
        
        # 최종 프로젝션
        out = self.projector(out)
        
        return out, alpha

class TriangleHead(nn.Module):
    def __init__(self, node_dim=256):
        super().__init__()
        self.common_mlp = nn.Sequential(
            nn.Linear(node_dim * 3, 512),
            nn.SiLU()
        )
        
        # 가중치 (0~1: 이 삼각형을 믿을 수 있는가?)
        self.weight_layer = nn.Linear(512, 1)
        
        # 법선 벡터 (3D: 이 삼각형의 기울기는 어떠한가?)
        self.normal_layer = nn.Linear(512, 3) 

    def forward(self, tri_feat):
        x = self.common_mlp(tri_feat)
        
        w_j = torch.sigmoid(self.weight_layer(x))
        n_j = torch.tanh(self.normal_layer(x)) # -1 ~ 1 사이 벡터
        n_j = F.normalize(n_j, dim=-1)         # 단위 벡터로 정규화
        
        return w_j, n_j
    

class GeoVOModel(nn.Module):
    def __init__(self, sigma_voting=2.0):
        super().__init__()
        self.gat = GeometricGAT(desc_dim=256)
        self.tri_head = TriangleHead(node_dim=256)
        self.sigma_voting = sigma_voting

    def forward(self, x, kpts, pts_3d_t, pts_3d_tm1, edge_index, tri_indices, focal, cx):
        """
        x: [N, 256] 특징점 디스크립터
        kpts: [N, 2] 2D 좌표
        pts_3d_t: [N, 3] 현재 프레임 3D 점
        pts_3d_tm1: [N, 3] 이전 프레임 3D 점
        """
        # 1. GAT & Triangle 피처 추출
        # node_out: [N, 256]
        node_out, _ = self.gat(x, kpts, pts_3d_t, edge_index)
        
        # 삼각형 정점 피처 결합 [T, 768]
        f_tri = torch.cat([
            node_out[tri_indices[:, 0]], 
            node_out[tri_indices[:, 1]], 
            node_out[tri_indices[:, 2]]
        ], dim=-1)
        
        # weights(w_j): [T, 1], pred_normals: [T, 3]
        weights, pred_normals = self.tri_head(f_tri)

        # 2. 모든 삼각형에 대한 개별 K_j 계산 [T, 3, 3]
        K_j = self.compute_individual_Kj(tri_indices, pts_3d_t, pts_3d_tm1)

        # 3. 삼각형별 로컬 R_j 후보군 추출 (Batch SVD)
        # 각 삼각형이 주장하는 개별적인 회전 행렬들
        R_j_candidates = self.batch_svd(K_j) # [T, 3, 3]
        
        # 4. 소실점 계산 및 Soft-Voting
        # 식 (13): xv = f * (r13 / r33) + cx
        r13 = R_j_candidates[:, 0, 2]
        r33 = R_j_candidates[:, 2, 2] + 1e-8 # zero division 방지
        xv_j = focal * (r13 / r33) + cx
        
        # 미분 가능한 방식으로 대표 소실점 xv_star 결정
        xv_star = differentiable_voting(xv_j, weights, sigma=self.sigma_voting)

        # 5. 소실점 일치도에 따른 Soft-Inlier 가중치(s_j) 계산
        # 예측된 소실점 위치와 각 삼각형의 소실점 위치가 가까울수록 높은 가중치
        dist_sq = (xv_j - xv_star)**2
        s_j = torch.exp(-dist_sq / (2 * self.sigma_voting**2)).unsqueeze(-1)

        # 6. 최종 Weighted SVD (최종 R 결정)
        # GAT의 시각적 신뢰도(weights) * 기하적 일관성(s_j)
        final_weights = weights * s_j
        R_final = estimate_rotation_svd_differentiable(final_weights, tri_indices, pts_3d_t, pts_3d_tm1)

        return R_final, pred_normals, final_weights

    def compute_individual_Kj(self, tri_indices, pts_t, pts_tm1):
        """각 삼각형별 3x3 상관 행렬 계산"""
        P_t = pts_t[tri_indices]     # [T, 3, 3]
        P_tm1 = pts_tm1[tri_indices] # [T, 3, 3]
        
        # 중심 정규화
        P_t_c = P_t - P_t.mean(dim=1, keepdim=True)
        P_tm1_c = P_tm1 - P_tm1.mean(dim=1, keepdim=True)
        
        # [T, 3, 3] = [T, 3, 3] @ [T, 3, 3]
        return torch.matmul(P_t_c.transpose(-2, -1), P_tm1_c)

    def batch_svd(self, K_j):
        """삼각형 개수(T)만큼 병렬로 SVD 수행"""
        # K_j: [T, 3, 3]
        U, S, Vh = torch.linalg.svd(K_j)
        V = Vh.mtranspose(-2, -1)
        R_j = torch.matmul(V, U.transpose(-2, -1))
        
        # Determinant 보정 (Batch 단위)
        det = torch.linalg.det(R_j)
        d = torch.ones((K_j.size(0), 3), device=K_j.device)
        d[:, 2] = torch.sign(det)
        D = torch.diag_embed(d)
        
        return V @ D @ U.transpose(-2, -1)
    
def estimate_rotation_svd_differentiable(weights, tri_indices, pts_3d_t, pts_3d_tm1):
        # 1. 데이터 준비 (기존 동일)
        P_t = pts_3d_t[tri_indices] 
        P_tm1 = pts_3d_tm1[tri_indices]
        
        P_t_centered = P_t - P_t.mean(dim=1, keepdim=True)
        P_tm1_centered = P_tm1 - P_tm1.mean(dim=1, keepdim=True)

        # 2. K_j 및 K_total 계산
        K_j = torch.matmul(P_t_centered.transpose(-2, -1), P_tm1_centered)
        
        # [주의] 미세한 노이즈(eps)를 더해 SVD 발산 방지
        K_total = torch.sum(weights.view(-1, 1, 1) * K_j, dim=0)
        K_total = K_total + torch.eye(3, device=K_total.device) * 1e-6 

        # 3. 미분 가능한 SVD
        U, S, Vh = torch.linalg.svd(K_total)
        V = Vh.mtranspose(-2, -1)
        
        # 4. Det 보정 로직 (미분 가능하게 수정)
        # torch.no_grad()를 제거하고 torch.det를 사용하여 부드럽게 연결
        det = torch.linalg.det(torch.matmul(V, U.transpose(-2, -1)))
        
        # Reflection(거울 반전) 방지용 대각 행렬 생성
        # det가 -1이면 마지막 열의 부호를 바꿈
        d = torch.ones(3, device=weights.device)
        d[2] = torch.sign(det) 
        D = torch.diag(d)

        # R = V @ D @ U^T
        R_final = V @ D @ U.transpose(-2, -1)

        return R_final

def differentiable_voting(xv_j, weights, img_width=1216, sigma=2.0):
    # 1. Grid 생성 (0, 1, 2, ..., W-1)
    grid = torch.arange(img_width, device=xv_j.device).float() # [W]
    
    # 2. Gaussian Voting (삼각형별로 가우시안을 뿌려 합산)
    # [T, 1] - [1, W] -> [T, W]
    diff_sq = (xv_j.unsqueeze(1) - grid.unsqueeze(0))**2
    voting_map = torch.sum(weights * torch.exp(-diff_sq / (2 * sigma**2)), dim=0) # [W]
    
    # 3. Soft-argmax (최종 소실점 위치 추정)
    # 온도 파라미터(10.0)를 높일수록 argmax와 비슷해지면서 미분 가능 유지
    probs = torch.softmax(voting_map * 10.0, dim=0) 
    xv_star = torch.sum(probs * grid)
    
    return xv_star
    


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
    def __init__(self, node_dim=256, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 입력: node_feat(256) + residual(2) + tri_weight(1) + vp_conf(1) = 260차원
        input_dim = node_dim + 2 + 1 + 1

        self.spatial_gat = GeometricGAT(in_channels=input_dim, out_channels=hidden_dim)
        self.norm_gat = nn.LayerNorm(hidden_dim)
        
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(),
            nn.Linear(128, 2) 
        )
        
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.SiLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, h, node_feat, r, tri_w, vp_s, edges, edge_attr):        
        """
        h: [B, N, hidden_dim] - 이전 hidden state
        node_feat: [B, N, 256] - GAT 노드 피처
        r: [B, N, 2] - 현재 재투영 오차 (Residual)
        tri_w: [B, N, 1] - 삼각형 가중치 (정점별 할당)
        vp_s: [B, N, 1] - 소실점 일관성 점수 (s_j)
        edges, edge_attr: 그래프 구조
        """
        B, N, _ = node_feat.shape
        device = node_feat.device

        # 1. 기하학적 정보 주입 (Feature Fusion)
        x_fused = torch.cat([node_feat, r, tri_w, vp_s], dim=-1) # [B, N, 260]
        x_flat = x_fused.view(-1, x_fused.size(-1))

        # 2. 그래프 데이터 병합 (배치 처리)
        flat_edges_list = [edges[i] + i * N for i in range(B)]
        edges_combined = torch.cat(flat_edges_list, dim=1)

        # 3. Spatial GAT (주변 오차 전파)
        x_spatial_flat, _ = self.spatial_gat(x_flat, edges_combined, edge_attr)
        x_spatial = self.norm_gat(x_spatial_flat.view(B, N, -1))

        # 4. GRU Update
        h_flat = h.view(-1, self.hidden_dim)
        x_in_flat = x_spatial.view(-1, self.hidden_dim)
        
        h_new_flat = self.gru(x_in_flat, h_flat)
        h_new = self.norm_h(h_new_flat.view(B, N, -1))

        # 5. Output Heads
        w_raw = self.weight_head(h_new)
        conf = torch.sigmoid(w_raw[..., 0:1]) 
        
        a_p = (self.alpha_pose_head(h_new).mean(dim=1) * 0.1) + 1e-4
        a_d = (self.alpha_depth_head(h_new) * 0.1) + 1e-4
        return h_new, conf, a_p, a_d
    


    
    
class GeometricBottleneck(nn.Module):
    def __init__(self, visual_dim=768, geo_dim=64, out_dim=256):
        super().__init__()
        
        # 1. 시각적 특징 압축 (Visual Compression)
        # 3C (768) -> 128로 줄여서 중복 정보 제거
        self.visual_enc = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )
        
        # 2. 기하학적 특징 확장 (Geometric Expansion)
        # 단순 수치가 아닌 '공간적 문맥'으로 변환
        self.geo_enc = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Linear(64, 128)
        )
        
        # 3. 최종 통합 (Feature Fusion)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def forward(self, v_feat, g_feat):
        # v_feat: [B, N, 768] (nodes_L + stereo_f + temporal_f)
        # g_feat: [B, N, 5] (inv_d, c_s, dx, dy, c_t)
        
        v_emb = self.visual_enc(v_feat)
        g_emb = self.geo_enc(g_feat)
        
        # 시각 정보와 기하 정보가 128:128로 대등하게 만남
        fused = torch.cat([v_emb, g_emb], dim=-1)
        return self.fusion(fused)
    










    
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




class EpipolarCrossAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.dim = feature_dim
        
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.merge = nn.Linear(feature_dim, feature_dim)

    def forward(self, nodes_L, nodes_R, kpts_L, kpts_R, return_attn=True):
        """
        nodes_L: [B, N, C]
        nodes_R: [B, M, C]
        kpts_L: [B, N, 2]
        kpts_R: [B, M, 2]
        """
        # --- 차원 방어 시작 ---
        if nodes_L.dim() == 2:
            nodes_L = nodes_L.unsqueeze(0) # [N, C] -> [1, N, C]
            nodes_R = nodes_R.unsqueeze(0)
            kpts_L = kpts_L.unsqueeze(0)   # [N, 2] -> [1, N, 2]
            kpts_R = kpts_R.unsqueeze(0)
        # --- 차원 방어 끝 ---
        B, N, C = nodes_L.shape
        M = nodes_R.shape[1]

        # 1. Linear Projection (배치 차원 유지)
        Q = self.q_proj(nodes_L)  # [B, N, C]
        K = self.k_proj(nodes_R)  # [B, M, C]
        V = self.v_proj(nodes_R)  # [B, M, C]

        # 2. 에피폴라 제약 설정 (Batch 대응 인덱싱)
        # kpts_L[:, :, 1:2] -> [B, N, 1]
        # kpts_R[:, :, 1].unsqueeze(1) -> [B, 1, M]
        dist_v = torch.abs(kpts_L[:, :, 1:2] - kpts_R[:, :, 1].unsqueeze(1)) # [B, N, M]
        dist_u = kpts_L[:, :, 0:1] - kpts_R[:, :, 0].unsqueeze(1)            # [B, N, M]
        
        # y축 차이가 적고(수평선), x축 차이가 양수(우측 카메라가 더 좌측에 투영됨)인 구간 마스크
        mask = (dist_v < 3.0) & (dist_u > 0) & (dist_u < 192)

        # 3. Batch Matrix Multiplication & Masking
        # K.transpose(1, 2)를 사용해 B를 제외한 N, M 차원만 연산
        attn = torch.matmul(Q, K.transpose(1, 2)) / (self.dim ** 0.5)   # [B, N, M]
        attn = attn.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn, dim=-1) # [B, N, M]                      
        
        # 4. 특징 및 시차(Disparity) 집계
        matched_features = torch.matmul(attn_weights, V) # [B, N, C]        
        
        # u_L - u_R 가중 평균
        init_disparity = torch.sum(attn_weights * dist_u, dim=-1, keepdim=True) # [B, N, 1]
        confidence = mask.any(dim=-1, keepdim=True).float()

        out = self.merge(matched_features)
        if return_attn:
            return out, init_disparity, confidence, attn_weights
        return out, init_disparity, confidence   
     
class StereoDepthModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 기하학적 정보를 위한 작은 인코더 추가
        self.geo_enc = nn.Sequential(
            nn.Linear(feature_dim + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 32)
        )

    def forward(self, nodes_L, matched_features, init_disp, confidence, intrinsics, baseline):
        """
        intrinsics: [B, 4] (fx, fy, cx, cy)
        baseline: 스칼라 (예: 0.54m)
        """
        B, N, _ = nodes_L.shape
        
        # 1. fx 추출 (intrinsics의 첫 번째 열)
        fx = intrinsics[:, 0:1] # [B, 1]
        fB = fx * baseline      # [B, 1]
        
        # 2. 브로드캐스팅을 위한 확장 [B, 1, 1]
        fB_expanded = fB.unsqueeze(-1) 
        
        # 3. Disparity를 Inverse Depth로 변환 (안전한 나눗셈)
        # fB / depth = disparity -> 1 / depth = disparity / fB
        inv_depth = torch.clamp(init_disp / (fB_expanded + 1e-8), min=1e-4, max=1.0)
        depth = 1.0 / (inv_depth + 1e-8)
        
        # 4. 특징 융합 (Visual + Geometric)
        # geo_input: [B, N, C + 1(inv_depth) + 1(confidence)]
        geo_input = torch.cat([nodes_L, inv_depth, confidence], dim=-1)
        geo_feat = self.geo_enc(geo_input) # [B, N, 32]
        
        # 최종 특징량: 원본 + 매칭된 상대 특징 + 기하학적 깊이 특징
        updated_nodes = torch.cat([nodes_L, matched_features, geo_feat], dim=-1)
        
        return updated_nodes, depth
    
class TemporalCrossAttention(nn.Module):
    def __init__(self, q_dim, kv_dim):
        super().__init__()
        self.q_dim = q_dim   # 544 (v_stereo_feat)
        self.kv_dim = kv_dim # 256 (f_Lt1)
        
        # Query는 544에서 변환
        self.q_proj = nn.Linear(q_dim, 256) # 결과는 256으로 통일
        # Key, Value는 256에서 변환
        self.k_proj = nn.Linear(kv_dim, 256)
        self.v_proj = nn.Linear(kv_dim, 256)
        
        self.merge = nn.Linear(256, 256)
        self.temp_geo_enc = nn.Sequential(
            nn.Linear(3, 32), # flow_residual(2) + confidence(1)
            nn.SiLU(),
            nn.Linear(32, 32)
        )

    def forward(self, nodes_t, nodes_t1, kpts_t1_pred, kpts_t1_actual, iter_idx):
        """
        nodes_t: [B, N, C]
        nodes_t1: [B, M, C]
        kpts_t1_pred: [B, N, 2]
        kpts_t1_actual: [B, M, 2]
        iter_idx: int
        """
        # --- 차원 방어 시작 ---
        if nodes_t.dim() == 2:
            nodes_t = nodes_t.unsqueeze(0)
            nodes_t1 = nodes_t1.unsqueeze(0)
            kpts_t1_pred = kpts_t1_pred.unsqueeze(0)
            kpts_t1_actual = kpts_t1_actual.unsqueeze(0)
        # --- 차원 방어 끝 ---
        B, N, C = nodes_t.shape
        M = nodes_t1.shape[1]

        # 1. Iteration 기반 윈도우 사이즈 결정
        if iter_idx == 0:
            win_size = 64.0
        else:
            win_size = max(4.0, 32.0 / (2 ** (iter_idx - 1)))
        
        # 2. Linear Projection (Batch 차원 유지)
        Q = self.q_proj(nodes_t)   # [B, N, C]
        K = self.k_proj(nodes_t1)  # [B, M, C]
        V = self.v_proj(nodes_t1)  # [B, M, C]

        # 3. 로컬 윈도우 마스크 생성 (Batch 대응)
        # kpts_t1_pred[:, :, 0:1] -> [B, N, 1]
        # kpts_t1_actual[:, :, 0].unsqueeze(1) -> [B, 1, M]
        # 결과: [B, N, M]
        dist_u = torch.abs(kpts_t1_pred[:, :, 0:1] - kpts_t1_actual[:, :, 0].unsqueeze(1))
        dist_v = torch.abs(kpts_t1_pred[:, :, 1:2] - kpts_t1_actual[:, :, 1].unsqueeze(1))

        mask = (dist_u < win_size) & (dist_v < win_size)

        # 4. Batch Matrix Multiplication (BMM)
        # Q: [B, N, C], K.transpose(1, 2): [B, C, M]
        attn = torch.matmul(Q, K.transpose(1, 2)) / (256 ** 0.5)
        attn = attn.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn, dim=-1)      # [B, N, M]

        # 5. 특성 및 좌표 집계
        # V: [B, M, C] -> matched_features: [B, N, C]
        matched_features = torch.matmul(attn_weights, V)
        # kpts_t1_actual: [B, M, 2] -> matched_kpts: [B, N, 2]
        matched_kpts = torch.matmul(attn_weights, kpts_t1_actual)

        # 6. Residual 및 Confidence 계산
        flow_residual = matched_kpts - kpts_t1_pred
        confidence = mask.any(dim=-1, keepdim=True).float() # [B, N, 1]
        flow_residual = flow_residual * confidence
        

        # forward 리턴 직전
        geo_temp = self.temp_geo_enc(torch.cat([flow_residual, confidence], dim=-1))
        return self.merge(matched_features), geo_temp, flow_residual, confidence


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
        