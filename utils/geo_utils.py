import torch
import numpy as np
from scipy.spatial import Delaunay

def compute_delaunay_edges(points):
    if torch.is_tensor(points):
        pts = points.detach().cpu().numpy()
    else:
        pts = points

    tri = Delaunay(pts)

    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted((simplex[0], simplex[1]))))
        edges.add(tuple(sorted((simplex[1], simplex[2]))))
        edges.add(tuple(sorted((simplex[2], simplex[0]))))
    edge_list = [[u, v] for u, v in edges] + [[v, u] for u, v in edges]

    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()