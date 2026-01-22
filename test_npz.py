import numpy as np
import os
import torch

def verify_precomputed_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 1. 데이터 로드
    data = np.load(file_path)
    
    print(f"--- 📂 파일 정보: {os.path.basename(file_path)} ---")
    
    # 2. 각 키별 데이터 확인
    # node_features: [N, 258] (Descriptor 256 + Norm_Kpts 2)
    node_features = data['node_features']
    # edges: [2, E] (Source, Target indices)
    edges = data['edges']
    # edge_attr: [E, 1] (Euclidean distance)
    edge_attr = data['edge_attr']
    # kpts: [N, 2] (Original Image Coordinates)
    kpts = data['kpts']

    print(f"✅ Node Features: {node_features.shape} (dtype: {node_features.dtype})")
    print(f"✅ Edges        : {edges.shape} (dtype: {edges.dtype})")
    print(f"✅ Edge Attrs   : {edge_attr.shape} (dtype: {edge_attr.dtype})")
    print(f"✅ Keypoints    : {kpts.shape} (dtype: {kpts.dtype})")

    # 3. 데이터 무결성 검사
    print("\n--- 🔍 데이터 무결성 체크 ---")
    
    # 노드 피처의 마지막 2차원이 정규화된 좌표(0~1)인지 확인
    norm_coords = node_features[:, -2:]
    is_normalized = np.all((norm_coords >= 0) & (norm_coords <= 1))
    print(f"⭐ 좌표 정규화 여부 (0~1): {'PASS' if is_normalized else 'FAIL'}")

    # 에지 인덱스가 노드 개수를 초과하지 않는지 확인
    num_nodes = node_features.shape[0]
    if edges.size > 0:
        is_edge_valid = np.max(edges) < num_nodes
        print(f"⭐ 에지 인덱스 유효성  : {'PASS' if is_edge_valid else 'FAIL'}")
        
        # 실제 거리와 edge_attr이 일치하는지 샘플 확인
        sample_dist = np.linalg.norm(kpts[edges[0, 0]] - kpts[edges[1, 0]])
        print(f"⭐ 거리 계산 일치도    : {sample_dist:.4f} vs {edge_attr[0, 0]:.4f}")
    else:
        print("⚠️ 에지가 생성되지 않았습니다 (특징점이 너무 적을 수 있음)")

    print("-" * 40)

if __name__ == "__main__":
    # 테스트하고 싶은 파일 경로 하나를 지정하세요
    # 예: 00번 시퀀스의 첫 번째 좌측 이미지 전처리 파일
    SAMPLE_PATH = "/home/jnu-ie/kys/Geo-VO/gendata/precomputed/00/image_2/000000.npz"
    
    verify_precomputed_data(SAMPLE_PATH)

