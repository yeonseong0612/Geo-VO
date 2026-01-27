import torch
import numpy as np
from src.model import VO
from CFG.vo_cfg import vo_cfg as cfg

def test_vo_dimensions(model, cfg):
    print("🚀 VO Model Dimension & Selection Test Start...")
    
    # 1. 가상 데이터 생성 (Batch=2, Images=4, H=352, W=1216)
    B = 2
    dummy_imgs = torch.randn(B, 4, 3, 352, 1216).cuda()
    dummy_calib = torch.tensor([[700, 700, 600, 180]] * B).float().cuda()
    
    batch = {
        'imgs': dummy_imgs,
        'calib': dummy_calib
    }

    model.cuda()
    model.eval()

    with torch.no_grad():
        print("\n--- [Step 1: Extraction & Selection] ---")
        # forward 로직의 시작 부분 시뮬레이션
        V = 4
        imgs_stacked = dummy_imgs.view(B * V, 3, 352, 1216)
        k_all_raw, f_all_raw = model.extractor(imgs_stacked)
        print(f"Raw Extraction: Keypoints {k_all_raw.shape}, Descriptors {f_all_raw.shape}")
        
        # Selector 통과
        top_k = 128
        f_all, k_all, indices = model.selector(k_all_raw, f_all_raw, (352, 1216), top_k=top_k)
        print(f"After Selection: Keypoints {k_all.shape}, Descriptors {f_all.shape}")
        
        # ⚠️ 질문 1 테스트: 800개가 아닌 128개가 들어가는가?
        actual_n = k_all.shape[1]
        if actual_n == top_k:
            print(f"✅ Success: Model is using {actual_n} selected points.")
        else:
            print(f"❌ Error: Model is still carrying {actual_n} points (Expected {top_k}).")

        # ⚠️ 질문 2 테스트: 디스크립터가 128인가 256인가?
        actual_desc_dim = f_all.shape[-1]
        print(f"✅ Descriptor Dimension: {actual_desc_dim}")
        
        print("\n--- [Step 2: Splitting & Flow] ---")
        # 현재 코드의 view(B, V, 800, ...) 부분을 체크
        try:
            k_split = k_all.view(B, V, actual_n, 2)
            f_split = f_all.view(B, V, actual_n, actual_desc_dim)
            print(f"Splitting OK: k_split {k_split.shape}, f_split {f_split.shape}")
        except RuntimeError as e:
            print(f"❌ View Error: {e}")

        # 3. 전체 forward 실행 시 각 모듈의 입력 차원 확인을 위해 
        # 모델 내부 곳곳에 print(f_Lt.shape) 등을 임시로 넣고 실행해봅니다.
        output = model(batch, iters=1, mode='test')
        
        print("\n--- [Step 3: Final Output Check] ---")
        print(f"Number of iterative poses: {len(output['poses'])}")
        print(f"Final Pose Shape: {output['poses'][-1].data.shape}")

if __name__ == "__main__":
    # 1. 실제 모델 객체 생성
    # cfg 객체가 필요하므로 이전에 정의한 cfg를 넣어줍니다.
    my_vo_model = VO(cfg) 
    
    # 2. 테스트 함수 호출
    test_vo_dimensions(my_vo_model, cfg)