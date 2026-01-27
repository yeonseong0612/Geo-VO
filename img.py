import os
import cv2
import numpy as np
from tqdm import tqdm

def check_kitti_dimensions(data_root, sequences):
    print(f"{'Seq':<5} | {'Original (H, W)':<15} | {'After Crop':<15} | {'Resize Needed?'}")
    print("-" * 60)
    
    for seq in sequences:
        img_dir = os.path.join(data_root, seq, 'image_2')
        if not os.path.exists(img_dir): continue
        
        # 첫 번째 이미지 하나만 샘플링
        img_name = sorted(os.listdir(img_dir))[0]
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        
        H, W, _ = img.shape
        
        # 1. 사용자님의 크롭 로직 적용
        crop_y = H % 32
        post_crop_h = H - crop_y
        post_crop_w = 1216 # 코드의 :1216 반영
        
        # 2. 리사이즈 필요 여부 확인
        needs_resize = (post_crop_h != 352 or post_crop_w != 1216)
        resize_ratio_y = 352 / post_crop_h
        
        status = f"YES (Ratio Y: {resize_ratio_y:.3f})" if needs_resize else "NO"
        
        print(f"{seq:<5} | {f'({H}, {W})':<15} | {f'({post_crop_h}, {post_crop_w})':<15} | {status}")

if __name__ == "__main__":
    # 실제 데이터 경로로 수정하세요
    RAW_DATA_PATH = "/home/jnu-ie/Dataset/kitti_odometry/data_odometry_color/dataset/sequences"
    SEQUENCES = [f"{i:02d}" for i in range(11)] # 00~10
    
    check_kitti_dimensions(RAW_DATA_PATH, SEQUENCES)