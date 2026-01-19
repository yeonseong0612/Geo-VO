import sys
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 루트 디렉토리를 path에 추가하여 내부 모듈을 불러올 수 있게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. 작성하신 데이터셋 클래스와 collate_fn 임포트
# datasets 폴더 안에 dataset.py가 있다면 아래와 같이 임포트합니다.
from src.loader import DataFactory, collate_fn

def test_loader():
    class Config:
        # 루트 기준 경로 설정
        proj_home = './' 
        odometry_home = '/home/yskim/projects/vo-labs/data/kitti_odometry/' 
        color_subdir = 'datasets/sequences/'
        poses_subdir = 'poses/'
        calib_subdir = 'datasets/sequences/'
        traintxt = 'train.txt'
        trainsequencelist = ['00'] 

    cfg = Config()
    
    try:
        # 3. 데이터셋 및 로더 초기화
        # mode='train'이면 gendata/train.txt를 읽으러 갑니다.
        dataset = DataFactory(cfg, mode='train')
        
        # 셔플을 True로 해서 다양한 프레임이 나오는지 확인합니다.
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

        print(f"✅ 데이터 로드 성공! 전체 데이터 개수: {len(dataset)}")

        for batch in loader:
            images = batch['images']       # [B, 4, 3, H, W]
            rel_poses = batch['rel_poses'] # SE3 객체 [B, 7]
            intrinsics = batch['intrinsics']
            
            print("\n" + "="*30)
            print("📊 배치 데이터 리포트")
            print("="*30)
            print(f"이미지 텐서 크기: {images.shape} (Batch, Views, C, H, W)")
            print(f"상대 포즈 (Translation + Quat):\n{rel_poses.data}")
            print(f"카메라 파라미터 [fx, fy, cx, cy]:\n{intrinsics}")

            # 4. 첫 번째 배치의 4장 이미지 시각화
            # [4, 3, H, W] -> [4, H, W, 3] 변환
            vis_imgs = images[0].permute(0, 2, 3, 1).cpu().numpy()
            titles = ['Lt (Left t)', 'Rt (Right t)', 'Lt+1 (Left t+1)', 'Rt+1 (Right t+1)']

            plt.figure(figsize=(20, 5))
            for i in range(4):
                plt.subplot(1, 4, i+1)
                plt.imshow(vis_imgs[i])
                plt.title(titles[i], fontsize=12)
                plt.axis('off')
            
            plt.suptitle(f"Sequence: {batch['seqs'][0]} | Frame: {batch['imgnums'][0]}", fontsize=15)
            plt.tight_layout()
            plt.show()

            break # 한 배치만 확인하고 종료

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        import traceback
        traceback.print_exc() # 어디서 에러가 났는지 상세히 출력

if __name__ == "__main__":
    test_loader()