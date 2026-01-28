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
    tri_indices_list: List of [T_i, 3] tensors (배치별 삼각형 인덱스 리스트)
    B: Batch size
    N: Number of nodes per sample (예: 800)
    device: torch.device
    """
    all_edges = []
    
    for i in range(B):
        tris = tri_indices_list[i] # [T_i, 3]
        if tris.shape[0] == 0:
            continue
            
        # 1. 삼각형의 세 변 추출 (1-2, 2-3, 3-1)
        e1 = tris[:, [0, 1]]
        e2 = tris[:, [1, 2]]
        e3 = tris[:, [2, 0]]
        
        # 2. 모든 변 결합 [3*T_i, 2]
        edges_one_way = torch.cat([e1, e2, e3], dim=0)
        
        # 3. 양방향 엣지 생성 (GAT는 일반적으로 무방향 그래프를 선호)
        edges_bi = torch.cat([edges_one_way, edges_one_way.flip(dims=[1])], dim=0)
        
        # 4. 배치 오프셋 적용 (Batch indexing)
        # i번째 배치의 노드 번호에 i * N을 더해줌
        offset = i * N
        edges_bi = edges_bi + offset
        
        all_edges.append(edges_bi)
    
    if len(all_edges) == 0:
        # 삼각형이 하나도 없는 경우 빈 엣지 반환
        return torch.empty((2, 0), dtype=torch.long, device=device)

    # 5. 모든 배치의 엣지를 하나로 결합 [Total_E, 2] -> [2, Total_E]
    edge_index = torch.cat(all_edges, dim=0).t().contiguous()
    
    # 6. (선택) 중복 엣지 제거 (삼각형이 인접할 경우 변이 겹침)
    # GAT 연산 속도를 위해 중복을 제거하는 것이 좋습니다.
    edge_index = torch.unique(edge_index, dim=1)
    
    return edge_index

def get_edge_attributes(edge_index, kpts, B, N):
    """
    edge_index: [2, E] (오프셋이 적용된 상태)
    kpts: [B, N, 2] (u, v)
    """
    # kpts를 [B*N, 2]로 펼침
    kpts_flat = kpts.view(-1, 2)
    
    src, dst = edge_index[0], edge_index[1]
    
    # 상대 좌표 Δu, Δv
    rel_uv = kpts_flat[dst] - kpts_flat[src]
    
    # 유클리드 거리
    dist = torch.norm(rel_uv, dim=-1, keepdim=True)
    
    # [E, 3] 속성 생성
    edge_attr = torch.cat([rel_uv, dist], dim=-1)
    
    return edge_attr