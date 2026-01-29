import torch
import numpy as np
from scipy.spatial import Delaunay

def compute_delaunay_edges(kpts):
    if not isinstance(kpts, np.ndarray):
        kpts = np.array(kpts)
        
    if len(kpts) < 3:
        return np.zeros((2, 0))

    tri = Delaunay(kpts)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i + 1) % 3]
            edges.add(tuple(sorted((u, v))))
    
    edges_np = np.array(list(edges)).T
    return edges_np

def process_single_view(k_np, dt_func): # dt_func 인자를 추가합니다.
    """단일 뷰에 대한 DT 및 Attr 계산"""
    # 엣지 계산 (전달받은 dt_func 사용)
    e_np = dt_func(k_np) # [2, E]
    
    if e_np.shape[1] > 0:
        # edge_attr 계산 (dist, dx, dy)
        src_pts = k_np[e_np[0]]
        dst_pts = k_np[e_np[1]]
        diff = src_pts - dst_pts
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        attr_np = np.concatenate([dist, diff], axis=1).astype(np.float32)
        return e_np, attr_np
    return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 3), dtype=np.float32)


def tri_indices_to_edges(tri_indices_list, B, N, device):
    """
    tri_indices_list: List of [T_b, 3] tensors (배치별 삼각형 인덱스)
    B: 배치 사이즈
    N: 이미지당 최대 특징점 개수 (예: 800)
    """
    all_edges = []

    for b in range(B):
        tris = tri_indices_list[b]
        if tris.shape[0] == 0:
            continue
            
        # 1. 삼각형의 세 변을 추출 (v1-v2, v2-v3, v3-v1)
        e1 = tris[:, [0, 1]]
        e2 = tris[:, [1, 2]]
        e3 = tris[:, [2, 0]]
        
        # 2. 모든 변을 하나로 합침 [3*T, 2]
        edges = torch.cat([e1, e2, e3], dim=0)
        
        # 3. 양방향 그래프를 위해 대칭 엣지 추가 (v2-v1, v3-v2, v1-v3)
        edges_rev = edges[:, [1, 0]]
        edges = torch.cat([edges, edges_rev], dim=0)
        
        # 4. 배치 오프셋 적용 (b번째 배치의 점 번호에 b*N을 더함)
        edges = edges + b * N
        
        all_edges.append(edges)

    # 5. 모든 배치의 엣지를 하나로 합치고 중복 제거
    if len(all_edges) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
        
    all_edges = torch.cat(all_edges, dim=0) # [Total_E, 2]
    
    # 중복된 엣지 제거 (Delaunay 특성상 인접 삼각형이 변을 공유함)
    all_edges = torch.unique(all_edges, dim=0)
    
    # 6. GAT 입력 규격 [2, Total_E]로 변경
    return all_edges.t().contiguous().to(device)


def compute_individual_Kj(tri_indices, pts_t, pts_tp1):
        P_t = pts_t[tri_indices]     # [T, 3, 3]
        P_tp1 = pts_tp1[tri_indices] # [T, 3, 3]
        
        P_t_c = P_t - P_t.mean(dim=1, keepdim=True)
        P_tp1_c = P_tp1 - P_tp1.mean(dim=1, keepdim=True)

        k_j = torch.matmul(P_t_c.transpose(-2, -1), P_tp1_c)        # [T, 3, 3] = [T, 3, 3] @ [T, 3, 3]

        return k_j

def batch_svd(K):
    K_norm = torch.norm(K, dim=(-2, -1), keepdim=True) + 1e-8
    K_scaled = K / K_norm

    K_scaled = K_scaled + torch.randn_like(K_scaled) * 1e-5

    # 3. SVD
    U, S, Vh = torch.linalg.svd(K_scaled)

    R = torch.matmul(U, Vh)
    det = torch.linalg.det(R)
    
    d = torch.ones((K.size(0), 3), device=K.device)
    d[:, 2] = torch.where(det < 0, -1.0, 1.0)
    D = torch.diag_embed(d)

    return U @ D @ Vh

def differentiable_voting(xv_j, weights, img_width=1216, sigma=2.0):
    # 1. Grid 생성 (0, 1, 2, ..., W-1)
    grid = torch.arange(img_width, device=xv_j.device).float() # [W]
    
    # 2. Gaussian Voting (삼각형별로 가우시안을 뿌려 합산)
    # [T, 1] - [1, W] -> [T, W]
    diff_sq = (xv_j.unsqueeze(1) - grid.unsqueeze(0))**2
    voting_map = torch.sum(weights * torch.exp(-diff_sq / (2 * sigma**2)), dim=0) # [W]
    
    # 3. Soft-argmax (최종 소실점 위치 추정)
    probs = torch.softmax(voting_map * 5.0, dim=0) 
    xv_star = torch.sum(probs * grid)
    
    return xv_star

    
def estimate_rotation_svd_differentiable(weights, tri_indices, pts_3d_t, pts_3d_tp1):
    # 1. 데이터 준비 (기존 동일)
    P_t = pts_3d_t[tri_indices] 
    P_tp1 = pts_3d_tp1[tri_indices]

    # 2. 중심 정규화
    P_t_centered = P_t - P_t.mean(dim=1, keepdim=True)
    P_tp1_centered = P_tp1 - P_tp1.mean(dim=1, keepdim=True)

    # 3. K_j 및 가중합 K_total 계산
    K_j = torch.matmul(P_t_centered.transpose(-2, -1), P_tp1_centered)
    K_total = torch.sum(weights.view(-1, 1, 1) * K_j, dim=0)
    K_total = K_total / (torch.norm(K_total) + 1e-8) 
    K_total = K_total + torch.randn_like(K_total) * 1e-5

    # 4. SVD 수행
    U, S, Vh = torch.linalg.svd(K_total)
    
    R = U @ Vh
    det = torch.linalg.det(R)
    
    d = torch.ones(3, device=weights.device)
    # sign 함수는 미분 불가능 지점이 있어 where가 더 안전함
    d[2] = torch.where(det < 0, -1.0, 1.0) 
    D = torch.diag(d)

    # 6. 최종 R 추출 (순서 중요: U @ D @ Vh)
    # Row vector 방식 (P_tp1 = P_t @ R)의 정답 순서입니다.
    R_final = U @ D @ Vh

    return R_final

def matrix_to_quat(R):
    """
    R: [B, 3, 3] 회전 행렬 배치를 [B, 4] 쿼터니언 (x, y, z, w)으로 변환
    Robust algorithm: 대각 성분 중 최대값을 기준으로 분기하여 수치적 안정성 확보
    """
    B = R.shape[0]
    device = R.device
    
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # Trace 계산
    tr = m00 + m11 + m22
    
    # 결과를 담을 텐서 [B, 4]
    quat = torch.zeros((B, 4), device=device)

    # 케이스 1: tr > 0 (가장 일반적인 경우)
    mask1 = tr > 0
    if mask1.any():
        s = torch.sqrt(tr[mask1] + 1.0) * 2
        quat[mask1, 3] = 0.25 * s
        quat[mask1, 0] = (m21[mask1] - m12[mask1]) / s
        quat[mask1, 1] = (m02[mask1] - m20[mask1]) / s
        quat[mask1, 2] = (m10[mask1] - m01[mask1]) / s

    # 케이스 2: tr <= 0 일 때 대각 성분 중 최대값에 따라 분기
    mask_else = ~mask1
    if mask_else.any():
        # 각 배치별로 m00, m11, m22 중 최대값 찾기
        m00_e, m11_e, m22_e = m00[mask_else], m11[mask_else], m22[mask_else]
        
        # m00이 가장 큰 경우
        mask2 = (m00_e > m11_e) & (m00_e > m22_e)
        if mask2.any():
            idx = mask_else.nonzero()[mask2].squeeze(-1)
            s = torch.sqrt(1.0 + m00[idx] - m11[idx] - m22[idx]) * 2
            quat[idx, 3] = (m21[idx] - m12[idx]) / s
            quat[idx, 0] = 0.25 * s
            quat[idx, 1] = (m01[idx] + m10[idx]) / s
            quat[idx, 2] = (m02[idx] + m20[idx]) / s
            
        # m11이 가장 큰 경우
        mask3 = (m11_e > m22_e) & (~mask2)
        if mask3.any():
            idx = mask_else.nonzero()[mask3].squeeze(-1)
            s = torch.sqrt(1.0 + m11[idx] - m00[idx] - m22[idx]) * 2
            quat[idx, 3] = (m02[idx] - m20[idx]) / s
            quat[idx, 0] = (m01[idx] + m10[idx]) / s
            quat[idx, 1] = 0.25 * s
            quat[idx, 2] = (m12[idx] + m21[idx]) / s
            
        # m22가 가장 큰 경우
        mask4 = (~mask2) & (~mask3)
        if mask4.any():
            idx = mask_else.nonzero()[mask4].squeeze(-1)
            s = torch.sqrt(1.0 + m22[idx] - m00[idx] - m11[idx]) * 2
            quat[idx, 3] = (m10[idx] - m01[idx]) / s
            quat[idx, 0] = (m02[idx] + m20[idx]) / s
            quat[idx, 1] = (m12[idx] + m21[idx]) / s
            quat[idx, 2] = 0.25 * s

    # 쿼터니언 정규화 (수치 오차 제거)
    return quat / torch.norm(quat, dim=-1, keepdim=True)