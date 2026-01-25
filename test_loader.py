import os
import cv2
import torch
import matplotlib.pyplot as plt
from CFG.vo_cfg import vo_cfg as cfg


seq = '00'      # 테스트할 시퀀스
imgnum = 0      # 테스트할 프레임 번호

def test_single_load():
    # 2. 경로 구성 확인
    img_paths = [
        os.path.join(cfg.odometry_home, cfg.color_subdir, seq, 'image_2', f"{str(imgnum).zfill(6)}.png"), # Lt
        os.path.join(cfg.odometry_home, cfg.color_subdir, seq, 'image_3', f"{str(imgnum).zfill(6)}.png"), # Rt
        os.path.join(cfg.odometry_home, cfg.color_subdir, seq, 'image_2', f"{str(imgnum+1).zfill(6)}.png"), # Lt1
        os.path.join(cfg.odometry_home, cfg.color_subdir, seq, 'image_3', f"{str(imgnum+1).zfill(6)}.png")  # Rt1
    ]

    print(f"--- 🔍 경로 확인 ---")
    for i, p in enumerate(img_paths):
        exists = "✅ 존재함" if os.path.exists(p) else "❌ 파일 없음"
        print(f"Path {i}: {p} ({exists})")

    # 3. 로드 로직 실행
    imgs = []
    for path in img_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ 경고: {path} 로드 실패!")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # (H, W, C) -> (C, H, W) 변환
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        imgs.append(img_tensor)

    if len(imgs) == 4:
        # 4. 차원 병합 (Stack)
        stacked_imgs = torch.stack(imgs)
        
        print(f"\n--- 📊 차원(Dimension) 분석 ---")
        print(f"낱개 이미지 텐서 모양: {imgs[0].shape}") # [3, H, W]
        print(f"최종 데이터['imgs'] 모양: {stacked_imgs.shape}") # [4, 3, H, W]
        print(f"차원 의미: [View_Count(4), Channels(3), Height, Width]")
        
        # 5. 시각적 확인 (첫 번째 이미지 출력)
        plt.imshow(stacked_imgs[0].permute(1, 2, 0))
        plt.title(f"Loaded Image: {seq} - {imgnum}")
        plt.show()
    else:
        print("❌ 로드된 이미지가 4장이 아닙니다.")

if __name__ == "__main__":
    test_single_load()