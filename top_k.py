import torch
import cv2
import numpy as np
import os
import torch.nn as nn
from src.extractor import SuperPointExtractor 
from src.layer import DescSelector

def run_top_k_visualization(img_path, device="cuda"):
    # 1. 모델 로드 및 초기화
    extractor = SuperPointExtractor(device=device)
    selector = DescSelector(in_dim=256, out_dim=128).to(device)
    selector.eval()

    # 2. 이미지 로드
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: {img_path}를 찾을 수 없습니다.")
        return
    
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 3. SuperPoint 특징점 추출
    with torch.no_grad():
        kpts, descs = extractor(img_tensor)
        
        # 4. Scatter Max 기반 그리드 셀렉터 적용 (하단 96칸 + 중간 32칸)
        final_feat, final_kpts, indices = selector(kpts, descs, (img_bgr.shape[0], img_bgr.shape[1]), top_k=128)

    # 5. 그리드 가이드라인 그리기 (하단 집중형으로 수정)
    vis_img = img_bgr.copy()
    h, w = vis_img.shape[:2]
    grid_color = (0, 0, 255) # 빨간색 격자
    
    # 영역 구분선
    y_lines = [int(h * 0.2), int(h * 0.5)]
    for y in y_lines:
        cv2.line(vis_img, (0, y), (w, y), grid_color, 1)
    
    # --- 중간 영역 시각화 (8x4 = 32칸) ---
    for x in np.linspace(0, w, 9): # 가로 8칸
        cv2.line(vis_img, (int(x), y_lines[0]), (int(x), y_lines[1]), grid_color, 1)
    for y in np.linspace(y_lines[0], y_lines[1], 5): # 세로 4칸
        cv2.line(vis_img, (0, int(y)), (w, int(y)), grid_color, 1)

    # --- 하단 영역 시각화 (16x6 = 96칸) ---
    # 사용자님의 의견대로 스케일 정밀도를 위해 가장 촘촘하게 배치
    for x in np.linspace(0, w, 17): # 가로 16칸
        cv2.line(vis_img, (int(x), y_lines[1]), (int(x), h), grid_color, 1)
    for y in np.linspace(y_lines[1], h, 7): # 세로 6칸
        cv2.line(vis_img, (0, int(y)), (w, int(y)), grid_color, 1)

    # 6. 특징점 타점 (빨간색: 선택됨 / 초록색: 탈락)
    kpts_np = kpts[0].cpu().numpy()
    selected_idx = indices[0].cpu().numpy()
    
    for i, pt in enumerate(kpts_np):
        if i in selected_idx:
            # 선택된 128개: 빨간색 (BGR: 0, 0, 255), 크기 2
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 2, (255, 0, 0), -1)
        else:
            # 탈락한 점: 초록색 (BGR: 0, 255, 0), 크기 1
            cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

    output_path = "top_k_perspective_grid.png"
    cv2.imwrite(output_path, vis_img)
    print(f"==> 하단 집중형 그리드 샘플링 결과가 저장되었습니다: {output_path}")

if __name__ == "__main__":
    SAMPLE_IMG = "/home/yskim/projects/vo-labs/data/kitti_odometry/datasets/sequences/01/image_2/000054.png"
    run_top_k_visualization(SAMPLE_IMG)