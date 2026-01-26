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