import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 사용자 정의 클래스 임포트 (파일 이름에 맞춰 수정하세요)
from src.extractor import SuperPointExtractor
from src.matcher import LightGlueMatcher
from thirdparty.lightglue.utils import load_image
from thirdparty.lightglue.viz2d import plot_images, plot_matches

# 1. 하드웨어 및 모델 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 사용자 정의 클래스로 인스턴스 생성
# weights 경로는 사용자님의 환경에 맞게 자동으로 설정되어 있습니다.
extractor = SuperPointExtractor(device=device)
matcher = LightGlueMatcher(device=device)
def run_matching_test(img_path0, img_path1):
    # 1. 이미지 로드
    image0 = load_image(img_path0).to(device)
    image1 = load_image(img_path1).to(device)

    # 2. 차원 확인 및 배치 차원 강제 추가
    # 만약 [C, H, W]라면 [1, C, H, W]로 변환하여 4차원을 맞춥니다.
    if image0.ndim == 3:
        image0 = image0.unsqueeze(0)
    if image1.ndim == 3:
        image1 = image1.unsqueeze(0)

    # 3. 채널 수 강제 조정 (SuperPoint용 1채널 변환)
    if image0.shape[1] == 3:
        image0 = image0.mean(dim=1, keepdim=True)
    if image1.shape[1] == 3:
        image1 = image1.mean(dim=1, keepdim=True)

    # 4. 특징점 추출 (사용자 정의 Extractor 호출)
    kpts0, desc0 = extractor(image0)
    kpts1, desc1 = extractor(image1)
    
    # 5. 이제 안전하게 4차원 텐서에서 H, W 추출 가능
    # [1, 1, H, W] 구조이므로 언패킹이 정상 작동합니다.
    _, _, h0, w0 = image0.shape
    _, _, h1, w1 = image1.shape

    data0 = {
        'keypoints': kpts0,
        'descriptors': desc0,
        'image_size': torch.tensor([[w0, h0]], device=device).float()
    }
    data1 = {
        'keypoints': kpts1,
        'descriptors': desc1,
        'image_size': torch.tensor([[w1, h1]], device=device).float()
    }
    
    # 6. 매칭 수행
    pred = matcher(data0, data1)
    
    # 7. 결과 정리 및 시각화 준비
    matches = pred['matches'][0] 
    m_kpts0 = kpts0[0][matches[..., 0]].cpu()
    m_kpts1 = kpts1[0][matches[..., 1]].cpu()
    
    print(f"이미지 크기: {w0}x{h0}")
    print(f"최종 매칭된 쌍: {len(matches)}")
    print(f"연산에 사용된 레이어 수: {pred['stop_layer']}")

    # 8. 시각화
    plot_images([image0[0].cpu(), image1[0].cpu()])
    plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.5)

    plot_images([image0[0].cpu(), image1[0].cpu()])
    plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.5)
    
    # 제목 추가 (선택)
    plt.title(f"LightGlue Matches: {len(matches)} pairs")
    


    # [방법 B] 결과 이미지 파일로 저장 (가장 권장)
    save_path = "matching_result.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"시각화 결과가 '{save_path}'에 저장되었습니다.")
    
    plt.close() # 메모리 해제
    
    return matches, m_kpts0, m_kpts1
# KITTI 이미지 경로 테스트
img0 = "/home/yskim/projects/vo-labs/data/kitti_odometry/datasets/sequences/00/image_2/000000.png"
img1 = "/home/yskim/projects/vo-labs/data/kitti_odometry/datasets/sequences/00/image_3/000000.png"

run_matching_test(img0, img1)