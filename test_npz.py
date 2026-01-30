import os
import numpy as np
from tqdm import tqdm
from CFG.vo_cfg import vo_cfg as cfg

def check_data_sanity():
    base_dir = cfg.precomputed_dir
    seq_list = [s for s in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, s))]
    
    bad_files = []
    total_files = 0
    
    print(f"🕵️ 데이터 무결성 조사를 시작합니다... (대상: {len(seq_list)}개 시퀀스)")

    for seq in seq_list:
        seq_path = os.path.join(base_dir, seq)
        files = [f for f in os.listdir(seq_path) if f.endswith('.npz')]
        
        for f in tqdm(files, desc=f"Checking {seq}", leave=False):
            total_files += 1
            file_path = os.path.join(seq_path, f)
            
            try:
                data = np.load(file_path)
                issues = []

                # 1. NaN 또는 Inf 체크 (모든 키값 대상)
                for key in data.files:
                    if np.isnan(data[key]).any():
                        issues.append(f"NaN in {key}")
                    if np.isinf(data[key]).any():
                        issues.append(f"Inf in {key}")

                # 2. 3D Points(pts_3d)의 유효성 체크
                pts_3d = data['pts_3d']
                # 깊이(Z)가 0이거나 음수인 경우 (Bundle Adjustment 터지는 주범)
                if (pts_3d[:, 2] <= 0).any():
                    zero_depth_count = np.sum(pts_3d[:, 2] <= 0)
                    issues.append(f"Zero/Neg Depth ({zero_depth_count} pts)")

                # 3. 특징점(kpts) 좌표 범위 체크 (이미지 밖으로 나갔는지)
                # 이미지 크기 설정 (cfg 참고: 1216, 352)
                kpts = data['kpts']
                if (kpts[:, 0] < 0).any() or (kpts[:, 0] > 1216).any() or \
                   (kpts[:, 1] < 0).any() or (kpts[:, 1] > 352).any():
                    issues.append("Kpts out of bounds")

                # 4. 삼각형 인덱스(tri_indices) 범위 체크
                if 'tri_indices' in data.files and data['tri_indices'].size > 0:
                    if data['tri_indices'].max() >= len(kpts):
                        issues.append("Invalid tri_indices (index error)")

                if issues:
                    bad_files.append(f"{seq}/{f} -> {' | '.join(issues)}")

            except Exception as e:
                bad_files.append(f"{seq}/{f} -> Error loading file: {str(e)}")

    # 결과 리포트
    print("\n" + "="*60)
    print(f"📊 무결성 조사 요약")
    print(f" - 전체 조사 파일: {total_files}개")
    print(f" - 결함 발견 파일: {len(bad_files)}개")
    print("="*60)

    if bad_files:
        save_path = "data_integrity_report.txt"
        with open(save_path, "w") as out:
            for item in bad_files:
                out.write(item + "\n")
        print(f"🚨 결함 리스트가 '{save_path}'에 저장되었습니다. 내용을 확인하세요!")
    else:
        print("✅ 모든 데이터가 깨끗합니다! 모델 내부 수식을 점검해 봅시다.")

if __name__ == "__main__":
    check_data_sanity()