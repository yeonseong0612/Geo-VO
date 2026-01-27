import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
# 사용자님이 작성하신 DataFactory와 collate_fn이 이 파일에 있다고 가정하거나 import 하세요.
from src.loader import DataFactory, vo_collate_fn 

def test_data_pipeline():
    print("🚀 KITTI Real-time Data Pipeline Test Start...")
    
    # 1. 시뮬레이션을 위한 더미 설정 (실제 경로가 있으면 실제 cfg를 넣으셔도 됩니다)
    class DummyCfg:
        proj_home = "./"
        odometry_home = "./data"
        color_subdir = "sequences"
        poses_subdir = "poses"
        calib_subdir = "sequences"
        trainsequencelist = ["00"]
        traintxt = "train.txt"
        batch_size = 4
        num_cpu = 2

    cfg = DummyCfg()

    # 테스트용 gendata/train.txt 및 디렉토리 생성 (필요 시)
    os.makedirs("gendata", exist_ok=True)
    with open("gendata/train.txt", "w") as f:
        f.write("00 0\n00 1\n00 2") # 시퀀스 00의 0, 1, 2번 인덱스

    try:
        # 2. 데이터셋 및 로더 초기화
        # 실제 데이터가 없는 경우 에러가 날 수 있으므로, __getitem__ 내부를 
        # 더미 리턴으로 살짝 수정해서 구조만 확인하는 것이 좋습니다.
        dataset = DataFactory(cfg, mode='train')
        loader = DataLoader(
            dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True, 
            collate_fn=vo_collate_fn
        )

        # 3. 데이터 한 배치 뽑기
        batch = next(iter(loader))

        # 4. 검증 루틴
        print("\n" + "="*30)
        print("✅ Batch Validation Results:")
        print(f"1. Images Shape:   {batch['imgs'].shape}") 
        # 기대 결과: [Batch, 4, 3, 352, 1216]
        
        print(f"2. Rel Pose Shape: {batch['rel_pose'].shape}")
        # 기대 결과: [Batch, 7] (x, y, z, qx, qy, qz, qw)
        
        print(f"3. Calib Shape:    {batch['calib'].shape}")
        # 기대 결과: [Batch, 4] (fx, fy, cx, cy)
        
        print(f"4. Sequences:      {batch['seq']}")
        print(f"5. Image Numbers:  {batch['imgnum']}")
        print("="*30)

        # 5. 시각적 확인 (첫 번째 샘플의 Lt 이미지)
        img_to_show = batch['imgs'][0, 0].permute(1, 2, 0).cpu().numpy()
        plt.figure(figsize=(12, 4))
        plt.imshow(img_to_show)
        plt.title(f"Sequence: {batch['seq'][0]} | Index: {batch['imgnum'][0]} (Lt)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        print("\n💡 Tip: 실제 KITTI 데이터가 경로에 없으면 cv2.imread가 None을 반환합니다.")
        print("구조만 확인하려면 DataFactory의 __getitem__에서 imgs를 torch.randn으로 리턴해보세요.")

if __name__ == "__main__":
    test_data_pipeline()