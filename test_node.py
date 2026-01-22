import os
import numpy as np
from tqdm import tqdm

def check_npz_nodes(data_dir):
    node_counts = []
    file_list = []

    # 모든 npz 파일 찾기
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".npz"):
                file_list.append(os.path.join(root, file))

    print(f"총 {len(file_list)}개의 파일을 발견했습니다. 검사를 시작합니다...")

    for f in tqdm(file_list):
        data = np.load(f)
        # 'node_features'의 첫 번째 차원이 노드 개수입니다.
        nodes = data['node_features'].shape[0]
        node_counts.append(nodes)

    node_counts = np.array(node_counts)
    unique_counts = np.unique(node_counts)

    print("\n--- 검사 결과 ---")
    if len(unique_counts) == 1:
        print(f"✅ 모든 파일의 노드 개수가 {unique_counts[0]}개로 일정합니다!")
    else:
        print(f"⚠️ 노드 개수가 일정하지 않습니다! (총 {len(unique_counts)}종류의 크기 발견)")
        print(f"최소 노드 수: {node_counts.min()}")
        print(f"최대 노드 수: {node_counts.max()}")
        print(f"평균 노드 수: {node_counts.mean():.2f}")
        
        # 어떤 파일들이 다른지 예시 출력
        print("\n[발견된 노드 개수 분포 (상위 5개)]")
        for count in unique_counts[:5]:
            count_num = np.sum(node_counts == count)
            print(f"- {count}개 노드: {count_num}개 파일")

# 사용 예시: 전처리 데이터가 저장된 경로를 넣으세요.
SAVE_PATH = "/home/jnu-ie/kys/Geo-VO/geovo_prcomputed"
check_npz_nodes(SAVE_PATH)