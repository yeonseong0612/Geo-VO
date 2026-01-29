import numpy as np
import os

def inspect_vo_npz(file_path):
    if not os.path.exists(file_path):
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return

    print(f"\n" + "="*60)
    print(f"🔍 데이터 진단: {os.path.basename(file_path)}")
    print("="*60)

    data = np.load(file_path)
    mask = data['mask']
    pts_3d = data['pts_3d']
    kpts_t = data['kpts']
    kpts_tp1 = data['kpts_tp1']

    # 1. 마스크 통계
    true_count = np.sum(mask)
    print(f"[Mask Status] Valid Points: {true_count} / {len(mask)} ({true_count/len(mask)*100:.1f}%)")

    # 2. 3D 좌표 점검 (중요: Mask가 True인 점들만 검사)
    valid_pts_3d = pts_3d[mask]
    if len(valid_pts_3d) > 0:
        z_vals = valid_pts_3d[:, 2]
        print(f"\n[3D Depth (Z) - Valid Only]")
        print(f"   Min Z: {np.min(z_vals):.4f}m (음수가 나오면 안 됩니다!)")
        print(f"   Max Z: {np.max(z_vals):.4f}m")
        print(f"   Mean Z: {np.mean(z_vals):.4f}m")
        
        if np.any(z_vals <= 0):
            print("   ⚠️ ALERT: 마스크된 영역 안에 여전히 0 이하의 Depth가 존재합니다!")
    else:
        print("\n   ⚠️ ALERT: 유효한 마스크 데이터가 하나도 없습니다.")

    # 3. 대응점 이동 거리 점검 (Mask가 True인 점들만 검사)
    valid_kpts_t = kpts_t[mask]
    valid_kpts_tp1 = kpts_tp1[mask]
    
    if len(valid_kpts_t) > 0:
        dist = np.linalg.norm(valid_kpts_t - valid_kpts_tp1, axis=1)
        print(f"\n[Tracking Quality - Valid Only]")
        print(f"   Max Displacement: {np.max(dist):.2f} pixels (1000 이상이면 위험)")
        print(f"   Mean Displacement: {np.mean(dist):.2f} pixels")
    
    # 4. 전체 데이터 범위 (전체 배열에서 비정상적인 값 존재 여부)
    print(f"\n[Global Numeric Check]")
    print(f"   Raw pts_3d Min/Max: {np.min(pts_3d):.2f} / {np.max(pts_3d):.2f}")
    if np.isnan(pts_3d).any(): print("   ⚠️ ALERT: NaN detected in raw data!")

    # 5. 삼각형 인덱스 점검
    tri = data['tri_indices']
    print(f"\n[Triangles]")
    print(f"   Total Triangles: {len(tri)}")
    if len(tri) > 0:
        # 모든 인덱스가 mask=True인 곳을 가리키는지 확인
        invalid_tri = np.any(~mask[tri])
        if invalid_tri:
            print("   ⚠️ ALERT: 삼각형 인덱스가 마스크된(False) 점을 참조하고 있습니다!")
        else:
            print("   ✅ 모든 삼각형이 유효한 점들로만 구성되었습니다.")

if __name__ == "__main__":
    # 새로 생성한 npz 파일 중 하나를 선택하세요
    target_path = "gendata/precomputed/00/pair_000000_000001.npz"
    inspect_vo_npz(target_path)