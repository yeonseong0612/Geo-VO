import sys
import os

# 현재 test.py가 있는 폴더를 시스템 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import cv2
import os
from torchvision import transforms
from src.model import VO

def main():
    # 1. 이미지 경로 설정 (맥북 경로)
    img_path = "./img/L/000000.png" 
    
    if not os.path.exists(img_path):
        print(f"❌ 이미지를 찾을 수 없습니다: {img_path}")
        return

    # 2. 이미지 로드 및 전처리
    print(f"🔄 이미지 로드 중: {img_path}")
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # 모델 입력용 텐서 변환 [1, 3, H, W]
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(img_rgb).unsqueeze(0)
    print(f"✅ 입력 텐서 준비 완료: {input_tensor.shape}")
    
    # 3. VO 시스템 초기화
    print("🚀 VO 시스템 초기화 중...")
    try:
        vo_system = VO()
        print("✅ 모델 로드 성공!")
    except Exception as e:
        print(f"❌ 모델 초기화 실패: {e}")
        return

    # 4. 실행
    print("🏃 파이프라인 가동...")
    with torch.no_grad():
        try:
            # kpts, refined_desc, attn 순서로 리턴한다고 가정
            kpts, refined_desc, attn = vo_system.run(input_tensor)
            
            print("\n" + "="*30)
            print("🎉 테스트 성공!")
            print(f"📍 특징점(Keypoints) 개수: {len(kpts)}")
            print(f"💎 강화된 디스크립터 크기: {refined_desc.shape}")
            if attn is not None:
                print(f"🔗 GAT 연결(Edge) 개수: {attn.shape[0]}")
            print("="*30)
            
        except Exception as e:
            print(f"❌ 실행 중 오류 발생: {e}")

if __name__ == "__main__":
    main()