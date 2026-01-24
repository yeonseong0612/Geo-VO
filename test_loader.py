import torch
from torch.utils.data import DataLoader
import numpy as np
from src.loader import DataFactory, vo_collate_fn
from CFG.vo_cfg import vo_cfg

def test_loader():
    # 1. 설정 로드 (테스트용으로 배치 사이즈 2 설정)
    vo_cfg.batchsize = 2
    vo_cfg.precomputed_dir = "/home/jnu-ie/kys/Geo-VO/gendata/precomputed"
    
    print("--- 🚀 DataLoader Test 시작 ---")
    
    try:
        # 2. 데이터셋 및 로더 초기화
        dataset = DataFactory(vo_cfg, mode='train')
        loader = DataLoader(
            dataset, 
            batch_size=vo_cfg.batchsize, 
            shuffle=True, 
            collate_fn=vo_collate_fn,
            num_workers=0  # 디버깅을 위해 0으로 설정
        )
        
        # 3. 첫 번째 배치 가져오기
        batch = next(iter(loader))
        
        print(f"✅ 배치 로드 성공! (Batch Size: {vo_cfg.batchsize})")
        print("-" * 40)

        # 4. 차원 정밀 검사
        B = vo_cfg.batchsize
        errors = 0

        # [Check 1] Rel Pose
        if batch['rel_pose'].shape == (B, 7):
            print(f"  [PASS] Rel Pose: {batch['rel_pose'].shape}")
        else:
            print(f"  [FAIL] Rel Pose: Expected ({B}, 7), Got {batch['rel_pose'].shape}")
            errors += 1

        # [Check 2] Node Features (핵심: 4차원 여부)
        if batch['node_features'].shape == (B, 4, 800, 256):
            print(f"  [PASS] Node Features: {batch['node_features'].shape}")
        else:
            print(f"  [FAIL] Node Features: Expected ({B}, 4, 800, 256), Got {batch['node_features'].shape}")
            errors += 1

        # [Check 3] Edges (리스트 구조 및 크기)
        if isinstance(batch['edges'], list) and len(batch['edges']) == B * 4:
            avg_edges = sum([e.shape[1] for e in batch['edges']]) // (B * 4)
            print(f"  [PASS] Edges List: Size {len(batch['edges'])}, Avg Edges: {avg_edges}")
        else:
            print(f"  [FAIL] Edges: 리스트 크기가 {B*4}가 아님")
            errors += 1

        # [Check 4] Calibration
        if batch['calib'].shape == (B, 4):
            print(f"  [PASS] Calibration: {batch['calib'].shape}")
        else:
            print(f"  [FAIL] Calibration: Expected ({B}, 4), Got {batch['calib'].shape}")
            errors += 1

        print("-" * 40)
        if errors == 0:
            print("✨ 모든 데이터 로더 테스트를 통과했습니다! 모델 학습을 시작하셔도 좋습니다.")
        else:
            print(f"❌ {errors}개의 항목에서 정합성 오류가 발견되었습니다.")

    except Exception as e:
        print(f"❌ 테스트 도중 에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loader()