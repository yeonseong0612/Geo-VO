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