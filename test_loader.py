import torch
import numpy as np
import time
from src.model import VO  # VO 클래스가 정의된 파일 경로

class DummyConfig:
    def __init__(self):
        self.baseline = 0.54  # KITTI 기준
        self.max_kpts = 800

def run_vo_integration_test():
    # 1. 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 테스트 장치: {device}")

    # 2. 모델 초기화
    cfg = DummyConfig()
    model = VO(cfg).to(device)
    model.eval()

    # 3. 가상 입력 데이터 생성 (Inference Mode 기준)
    # [Batch, View, Channel, H, W] -> KITTI 해상도 (376, 1241)
    B, V, C, H, W = 1, 4, 3, 376, 1241
    dummy_imgs = torch.randn(B, V, C, H, W).to(device)
    
    # intrinsics: [fx, fy, cx, cy]
    dummy_calib = torch.tensor([[718.8, 718.8, 607.1, 185.2]]).to(device)

    batch = {
        'imgs': dummy_imgs,
        'calib': dummy_calib
    }

    print(f"📦 입력 데이터 준비 완료: {dummy_imgs.shape}")
    print("⚙️ 모델 추론 시작 (SP 추출 + Parallel DT + DBA Loop)...")

    # 4. 추론 실행 및 시간 측정
    start_time = time.time()
    try:
        with torch.no_grad():
            # iters=12 정도로 설정하여 최적화 루프 테스트
            outputs = model(batch, iters=12, mode='test')
        
        end_time = time.time()
        elapsed = end_time - start_time

        # 5. 결과 검증
        print("\n" + "="*30)
        print("✅ 테스트 성공!")
        print(f"⏱️ 소요 시간: {elapsed:.3f} 초")
        print(f"📍 포즈 리스트 길이: {len(outputs['poses'])} (iters와 일치해야 함)")
        
        # 마지막 이터레이션의 결과물 형태 확인
        last_pose = outputs['poses'][-1]
        last_depth = outputs['depths'][-1]
        
        print(f"🚗 최종 포즈 차원: {last_pose.data.shape}")  # [B, 7] (tx, ty, tz, qx, qy, qz, qw)
        print(f"💎 최종 깊이 차원: {last_depth.shape}")     # [B, 800]
        print("="*30)

    except Exception as e:
        print(f"\n❌ 테스트 중 오류 발생!")
        print(f"에러 메시지: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_vo_integration_test()