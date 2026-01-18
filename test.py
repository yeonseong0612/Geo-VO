import torch
import numpy as np
from scipy.spatial import Delaunay
from torch_geometric.nn import GATv2Conv

def test_full_pipeline():
    # 1. 랜덤 SuperPoint 좌표 (2D) 생성 (100개)
    points = np.random.rand(100, 2)
    descriptors = torch.randn(100, 128) # 디스크립터

    # 2. Delaunay Triangulation 수행
    tri = Delaunay(points)
    
    # 3. Edge Index 생성 (i -> j)
    edges = set()
    for s in tri.simplices:
        edges.add(tuple(sorted((s[0], s[1]))))
        edges.add(tuple(sorted((s[1], s[2]))))
        edges.add(tuple(sorted((s[2], s[0]))))
    
    edge_list = []
    for u, v in edges:
        edge_list.append([u, v])
        edge_list.append([v, u])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # 4. 가상의 기하학적 에러 (e_ij) 생성
    edge_attr = torch.randn(edge_index.size(1), 1)

    # 5. GAT 모델 통과
    model = GATv2Conv(in_channels=128, out_channels=128, edge_dim=1, heads=4, concat=False)
    
    try:
        out = model(descriptors, edge_index, edge_attr)
        print("✅ Delaunay + GAT 파이프라인 테스트 성공!")
        print(f"총 노드 수: {descriptors.size(0)}")
        print(f"총 엣지 수: {edge_index.size(1)}")
        print(f"결과 텐서 크기: {out.shape}")
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    test_full_pipeline() 