import sys
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. 경로 설정 (필요시 수정)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# DataFactory와 vo_collate_fn이 들어있는 파일명에 맞춰 임포트
from src.loader import DataFactory, vo_collate_fn 

class TestConfig:
    # --- 경로 설정 (사용자 환경에 맞게 수정) ---
    proj_home = './'
    odometry_home = '/home/yskim/projects/vo-labs/data/kitti_odometry/' # KITTI 데이터 루트
    precomputed_dir = './data/precomputed' # NPZ 저장 경로
    
    color_subdir = 'datasets/sequences/'
    poses_subdir = 'poses/'
    calib_subdir = 'datasets/sequences/'
    
    traintxt = 'train.txt'
    valtxt = 'val.txt'
    trainsequencelist = ['00']
    valsequencelist = ['09']

def run_test(mode='val'):
    cfg = TestConfig()
    print(f"\n[{mode.upper()} MODE] 테스트 시작...")
    
    try:
        # 1. 데이터셋 초기화
        dataset = DataFactory(cfg, mode=mode)
        
        # 2. 로더 초기화 (배치 사이즈 2로 가변 데이터 작동 확인)
        loader = DataLoader(
            dataset, 
            batch_size=1, 
            shuffle=True, 
            collate_fn=vo_collate_fn
        )

        print(f"데이터셋 로드 성공! (길이: {len(dataset)})")

        for i, batch in enumerate(loader):
            print(f"\n--- Batch {i+1} Report ---")
            print(f"Sequence: {batch['seq']}")
            print(f"Frame Indices: {batch['imgnum']}")
            print(f"Rel Pose Shape: {batch['rel_pose'].shape}") # [B, 7]
            print(f"Intrinsics (clib): {batch['clib'].shape}") # [B, 4]

            # --- [CASE A] Validation 모드 (이미지 확인) ---
            if mode == 'val':
                images = batch['images'] # [B, 4, 3, H, W]
                print(f"Images Shape: {images.shape}")
                
                # 첫 번째 샘플(배치 0번) 선택 및 차원 변환
                # [4, 3, H, W] -> [4, H, W, 3]
                # .permute(0, 2, 3, 1) 로 수정해야 합니다!
                vis_img = images[0].permute(0, 2, 3, 1).cpu().numpy()
                
                titles = ['Lt (Left t)', 'Rt (Right t)', 'Lt+1 (Left t+1)', 'Rt+1 (Right t+1)']
                plt.figure(figsize=(20, 5))
                for j in range(4):
                    plt.subplot(1, 4, j+1)
                    # vis_img[j]는 이제 [H, W, 3] 크기의 이미지가 됩니다.
                    plt.imshow(vis_img[j])
                    plt.title(titles[j])
                    plt.axis('off')
                
                plt.suptitle(f"Mode: {mode} | Seq: {batch['seq'][0]} | Frame: {batch['imgnum'][0]}")
                plt.tight_layout()
                plt.show()

            # --- [CASE B] Training 모드 (NPZ 확인) ---
            elif mode == 'train':
                # collate_fn에서 합쳐진 텐서들 확인
                print(f"Node Features: {batch['node_features'].shape}") # [B*4, max_n, 258]
                print(f"Edges: {batch['edges'].shape}")                 # [B*4, 2, max_e]
                print(f"Edge Attr: {batch['edge_attr'].shape}")         # [B*4, max_e, 1]
                print(f"Masks: {batch['masks'].shape}")                # [B*4, max_n]
                
                # 데이터 유효성 검사 (0이 아닌 값이 들어있는지)
                if torch.any(batch['node_features'] != 0):
                    print("Node features contain valid data.")
                if torch.any(batch['masks'] == True):
                    print(f"Valid nodes count in first sample: {batch['masks'][0].sum().item()}")

            if i >= 0: break # 한 배치만 확인

    except Exception as e:
        print(f"에러 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 1. 먼저 이미지가 잘 나오는지 확인
    run_test(mode='val')
    
    # 2. 전처리된 NPZ 데이터가 잘 묶이는지 확인 (NPZ 생성이 완료된 경우만)
    # run_test(mode='train')