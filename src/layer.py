import torch
import torch.nn as nn
import torch.nn.functional as F
from lietorch import SE3
from torch_geometric.nn import GATv2Conv

class GeometricGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=256, out_channels=256, heads=4):
        super().__init__()
        
        # 입력 차원과 히든 차원이 다를 경우 Residual을 위해 맞춰주는 레이어
        self.res_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

        self.conv1 = GATv2Conv(in_channels, hidden_channels // heads,
                               heads=heads, edge_dim=3)
        self.norm1 = nn.LayerNorm(hidden_channels)
        
        self.conv2 = GATv2Conv(hidden_channels, out_channels // heads, 
                               heads=heads, edge_dim=3, concat=True)
        self.norm2 = nn.LayerNorm(out_channels)
        
        self.projector = nn.Linear(out_channels, out_channels)
        self.SiLU = nn.SiLU()

    def forward(self, x, edge_index, edge_attr):
        # 1st Layer + Residual
        identity = self.res_proj(x) # 여기서 차원을 hidden_channels(256)로 강제 일치    
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = self.SiLU(x)
        x = x + identity # 이제 무조건 256 + 256이 됨
        
        # 2nd Layer + Residual
        identity = x # 이미 256차원
        x, (edge_index, alpha) = self.conv2(x, edge_index, edge_attr, return_attention_weights=True)
        x = self.norm2(x)
        x = self.SiLU(x)
        x = x + identity 
        
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
        inv_H_dd = 1.0 / (H_dd + 1e-6)
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

        return delta_pose.squeeze(-1), delta_depth
    
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
        new_pose = delta_SE3 * curr_pose

        return new_pose, new_depth
    

class GraphUpdateBlock(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.spatial_gat = GeometricGAT(in_channels=768, out_channels=hidden_dim)
        self.norm_gat = nn.LayerNorm(hidden_dim)
        
        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        
        self.weight_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 2)
        )
        
        # 4. Residual Head (Flow correction dx, dy)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 2)
        )
        
        # 5. Alpha Heads (업데이트 보폭 조절)
        self.alpha_pose_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(), 
            nn.Linear(128, 1), nn.Sigmoid()
        )
        self.alpha_depth_head = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.SiLU(), 
            nn.Linear(128, 1), nn.Sigmoid()
        )
        nn.init.constant_(self.weight_head[-1].bias, 0.0) 
        nn.init.xavier_normal_(self.weight_head[-1].weight, gain=0.01)

    def forward(self, h, flow_res, x_fused, edges, edge_attr):
        B, N, D = x_fused.shape
        device = x_fused.device

        # --- 1. 입력 준비 ---
        x = x_fused 
        x_flat = x.reshape(-1, x.size(-1)) # [BN, 256]
        # --- 2. 그래프 데이터 병합 (GPU 병목 최적화) ---
        # 매번 루프에서 tensor() 생성을 피하기 위해 미리 텐서인 경우만 처리
        flat_edges_list = []
        for i in range(B):
            e = edges[i] if isinstance(edges[i], torch.Tensor) else torch.tensor(edges[i], device=device)
            flat_edges_list.append(e + i * N)
        print(f"--- DEBUG ---")
        print(f"List length: {len(flat_edges_list)}")
        for idx, e in enumerate(flat_edges_list):
            print(f"Tensor {idx} shape: {e.shape}, dim: {e.dim()}")
        # -------------
        edges_combined = torch.cat(flat_edges_list, dim=1).to(device)
        edge_attr_combined = torch.cat(edge_attr, dim=0).to(device) if isinstance(edge_attr, list) else edge_attr
      
        # --- 3. Spatial GAT 연산 ---
        x_spatial_flat, _ = self.spatial_gat(x_flat, edges_combined, edge_attr_combined)
        x_spatial = x_spatial_flat.view(B, N, -1)
        x_spatial = self.norm_gat(x_spatial) # 수치 안정화용 노름

        # --- 4. GRU 연산 (히스토리 반영) ---
        h_flat_in = x_spatial.reshape(-1, self.hidden_dim) 
        h_prev_flat = h.reshape(-1, self.hidden_dim)
        
        h_new_flat = self.gru(h_flat_in, h_prev_flat)
        h_new = h_new_flat.view(B, N, -1)
        h_new = self.norm_h(h_new)

        # --- 5. Heads 출력 ---
        r = self.residual_head(h_new) 
        
        # w_raw: [B, N, 2] -> (Confidence, Scale)
        w_raw = self.weight_head(h_new)
        # weight: 0.05 ~ 1.0 (Safe Weight 적용)
        w = torch.sigmoid(w_raw[..., 0:1]) * 0.95 + 0.05
        # scale: DBA 내부에서 활용할 추가 댐핑용 (필요시)
        # 만약 이전처럼 w 하나만 쓰신다면 w_raw[..., 0:1]만 사용하시면 됩니다.

        # a_p, a_d: 업데이트 보폭 (0.1 수준으로 스케일 다운하여 발산 방지)
        a_p = (self.alpha_pose_head(h_new).mean(dim=1) * 0.1) + 1e-4 # [B, 1]
        a_d = (self.alpha_depth_head(h_new) * 0.1) + 1e-4           # [B, N, 1]

        return h_new, r, w, a_p, a_d



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