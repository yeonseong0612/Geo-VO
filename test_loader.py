import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from src.loader import DataFactory, vo_collate_fn
from CFG.vo_cfg import vo_cfg as cfg

# 기존에 작성하신 DataFactory, vo_collate_fn, vo_cfg가 임포트 가능하다고 가정합니다.
# from your_module import DataFactory, vo_collate_fn, cfg

def test_data_factory():
    print("🚀 DataFactory 검증을 시작합니다...")

    # 1. Train 모드 검증
    try:
        train_dataset = DataFactory(cfg, mode='train')
        train_loader = DataLoader(
            train_dataset, 
            batch_size=2, 
            shuffle=True, 
            collate_fn=vo_collate_fn
        )
        
        print(f"\n[Train Mode] 총 샘플 수: {len(train_dataset)}")
        
        # 첫 번째 배치 가져오기
        batch = next(iter(train_loader))
        
        print("✅ Train Batch 로드 성공!")
        print(f" - kpts shape: {batch['kpts'].shape} (Expected: [B, 800, 2])")
        print(f" - descs shape: {batch['descs'].shape} (Expected: [B, 800, 256])")
        print(f" - pts_3d shape: {batch['pts_3d'].shape} (Expected: [B, 800, 3])")
        print(f" - rel_pose shape: {batch['rel_pose'].shape} (Expected: [B, 7])")
        
        # 가변 길이 데이터 체크
        print(f" - temporal_matches (list) len: {len(batch['temporal_matches'])}")
        print(f" - 첫 번째 샘플 매칭 수: {batch['temporal_matches'][0].shape[0]}")
        
    except Exception as e:
        print(f"❌ Train Mode 에러 발생: {e}")

    print("-" * 50)

    # 2. Val 모드 검증 (이미지 로드)
    try:
        val_dataset = DataFactory(cfg, mode='val')
        val_loader = DataLoader(
            val_dataset, 
            batch_size=1, 
            shuffle=False, 
            collate_fn=vo_collate_fn
        )
        
        print(f"[Val Mode] 총 샘플 수: {len(val_dataset)}")
        
        val_batch = next(iter(val_loader))
        
        print("✅ Val Batch 로드 성공!")
        print(f" - imgs shape: {val_batch['imgs'].shape} (Expected: [B, 4, 3, 352, 1216])")
        
        # 이미지 시각화 테스트 (옵션)
        # sample_img = val_batch['imgs'][0, 0].permute(1, 2, 0).numpy()
        # plt.imshow(sample_img)
        # plt.title(f"Val Sample: Seq {val_batch['seq'][0]}")
        # plt.show()
        
    except Exception as e:
        print(f"❌ Val Mode 에러 발생: {e}")

    print("\n✨ 모든 검증이 완료되었습니다.")

if __name__ == "__main__":
    # 테스트 실행
    test_data_factory()