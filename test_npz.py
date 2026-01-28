import numpy as np
import os

def verify_pair_data(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return

    # 데이터 로드
    data = np.load(file_path)
    
    print(f"\n{'='*20} 📂 Pair 파일 정보: {os.path.basename(file_path)} {'='*20}")
    
    # 1. 저장된 항목별 Shape 및 타입 확인
    keys = ['kpts', 'pts_3d', 'descs', 'temporal_matches', 'match_scores', 'mask', 'tri_indices', 'K']
    for k in keys:
        if k in data:
            print(f"✅ {k:<18} : Shape {str(data[k].shape):<15} | dtype: {data[k].dtype}")
        else:
            print(f"⚠️ {k:<18} : 데이터가 존재하지 않습니다!")

    # 2. 기하학적 무결성 체크
    print(f"\n{'*'*20} 🔍 데이터 정밀 체크 {'*'*20}")
    
    # [A] 3D 점(pts_3d) 유효성 확인 [Image of 3D point cloud projection in stereo vision]
    pts_3d = data['pts_3d']
    z_values = pts_3d[:, 2] # Depth
    valid_z = z_values[z_values > 0]
    print(f"⭐ 유효 Depth(Z>0) 수 : {len(valid_z)} / {len(z_values)}")
    if len(valid_z) > 0:
        print(f"⭐ 평균 Depth 거리    : {np.mean(valid_z):.2f}m (Min: {np.min(valid_z):.1f}m, Max: {np.max(valid_z):.1f}m)")

    # [B] 시간적 매칭(Temporal Matches) 확인 [Image of feature matching between consecutive video frames]
    matches = data['temporal_matches']
    scores = data['match_scores']
    print(f"⭐ 시간적 매칭 쌍 수  : {len(matches)}개")
    if len(scores) > 0:
        print(f"⭐ 매칭 신뢰도 평균   : {np.mean(scores):.4f}")

    # [C] 삼각형 인덱스 유효성
    tri_idx = data['tri_indices']
    kpts_len = len(data['kpts'])
    if len(tri_idx) > 0:
        is_tri_safe = np.max(tri_idx) < kpts_len
        print(f"⭐ 삼각형 인덱스 안전 : {'PASS' if is_tri_safe else 'FAIL (Out of Bounds)'}")

    # [D] 주점 보정 값 (cy) 재확인
    K = data['K']
    print(f"⭐ 적용된 주점(cy)    : {K[1, 2]:.2f} (보정 여부 확인용)")

    print(f"{'='*60}\n")

if __name__ == "__main__":
    # 실제로 생성된 pair npz 파일 경로로 수정하세요
    SAMPLE_PAIR_PATH = "/home/jnu-ie/kys/Geo-VO/gendata/precomputed/00/pair_000000_000001.npz"
    verify_pair_data(SAMPLE_PAIR_PATH)