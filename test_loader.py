import torch
import numpy as np
from src.model import VO # 저장하신 파일 경로에 맞춰 수정하세요

class MockConfig:
    def __init__(self):
        self.baseline = 0.54  # KITTI 기준
        self.max_kpts = 800

def test_inference_mode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 테스트 시작 (Device: {device})")

    # 1. 모델 초기화
    cfg = MockConfig()
    model = VO(cfg).to(device)
    model.eval()

    # 2. 가상 배치 데이터 생성
    # [Batch, View(Lt,Rt,Lt1,Rt1), Channel, H, W]
    B, V, C, H, W = 1, 4, 3, 376, 1241
    dummy_imgs = torch.randn(B, V, C, H, W).to(device)
    
    # intrinsics: [fx, fy, cx, cy]
    dummy_calib = torch.tensor([[718.8, 718.8, 607.1, 185.2]]).to(device)

    batch = {
        'imgs': dummy_imgs,
        'calib': dummy_calib
    }

    print("📸 입력 이미지 준비 완료. 추론 실행 중...")

    # 3. 모델 실행
    try:
        with torch.no_grad():
            outputs = model(batch, iters=8, mode='test')
        
        # 4. 결과 출력 및 검증
        print("\n✅ 추론 성공!")
        print(f"📍 출력 결과물 키: {list(outputs.keys())}")
        
        last_pose = outputs['poses'][-1]
        print(f"🚗 추정된 상대 포즈 (마지막 iteration): \n{last_pose.data}")
        
        last_depth = outputs['depths'][-1]
        print(f"💎 추정된 깊이 맵 shape: {last_depth.shape}") # [B, N]
        
        # 가중치 확인 (네트워크가 얼마나 확신하는지)
        last_weight = outputs['weights'][-1]
        print(f"⚖️ 매칭 가중치 평균: {last_weight.mean().item():.4f}")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_mode()