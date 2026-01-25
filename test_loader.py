import torch
from torch.utils.data import DataLoader
import numpy as np
from src.loader import DataFactory, vo_collate_fn, vo_test_collate_fn
from src.model import VO # 모델 클래스 임포트
from CFG.vo_cfg import vo_cfg
from lietorch import SE3

@torch.no_grad()
def run_integrated_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 0. 공통 설정
    vo_cfg.batchsize = 1 # 추론 테스트를 위해 1로 설정
    model = VO(vo_cfg).to(device)
    model.eval()

    print(f"--- 🚀 Integrated System Test (Device: {device}) ---")

    # ==========================================================
    # PART 1. 학습 모드 테스트 (Precomputed 데이터)
    # ==========================================================
    print("\n[STEP 1] Training Mode Data Check (.npz)")
    train_ds = DataFactory(vo_cfg, mode='train')
    train_loader = DataLoader(train_ds, batch_size=vo_cfg.batchsize, 
                              shuffle=True, collate_fn=vo_collate_fn)
    
    train_batch = next(iter(train_loader))
    print(f"✅ Train Batch Keys: {list(train_batch.keys())}")
    print(f"✅ Node Features Shape: {train_batch['node_features'].shape}") # [B, 4, 800, 256]

    # ==========================================================
    # PART 2. 추론 모드 테스트 (Raw Images + Real-time SP/DT)
    # ==========================================================
    print("\n[STEP 2] Inference Mode Data Check (Raw Images)")
    test_ds = DataFactory(vo_cfg, mode='test')
    test_loader = DataLoader(test_ds, batch_size=vo_cfg.batchsize, 
                             shuffle=False, collate_fn=vo_test_collate_fn)
    
    test_batch = next(iter(test_loader))
    
    # 데이터 장비 이동
    for k in test_batch:
        if isinstance(test_batch[k], torch.Tensor):
            test_batch[k] = test_batch[k].to(device)

    print(f"✅ Test Batch Keys: {list(test_batch.keys())}")
    print(f"✅ Raw Images Shape: {test_batch['imgs'].shape}") # [B, 3, 3, H, W]

    # ==========================================================
    # PART 3. 최종 추론 실행 및 결과 산출 (The Moment of Truth)
    # ==========================================================
    print("\n[STEP 3] Full Inference Execution")
    try:
        # 모델 통과 (SuperPoint 추출 및 DT 그래프 생성이 내부에서 일어남)
        outputs = model(test_batch, iters=12, mode='test')
        
        pred_poses = outputs['poses'][-1] # 마지막 이터레이션 결과 [B, 7]
        gt_poses = SE3(test_batch['rel_pose']) # [B, 7]
        
        # 오차 계산
        diff = pred_poses * gt_poses.inv()
        v = diff.log() # [B, 6] -> [tx, ty, tz, rx, ry, rz]
        
        t_err = v[:, :3].norm(dim=-1).mean().item()
        r_err = v[:, 3:].norm(dim=-1).mean().item()

        print("-" * 40)
        print(f"📊 프레임 번호: {test_batch['imgnum'][0]}")
        print(f"📍 Translation Error: {t_err:.4f} m")
        print(f"🔄 Rotation Error:    {r_err:.4f} rad")
        print("-" * 40)
        
        if t_err < 1.0: # 1미터 미만이면 일단 성공적으로 작동하는 것으로 판단
            print("✨ 결과: 모델이 이미지로부터 포즈를 성공적으로 추정했습니다!")
        else:
            print("⚠️ 경고: 오차가 큽니다. 가중치나 전처리를 확인하세요.")

    except Exception as e:
        print(f"❌ 추론 도중 치명적 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_integrated_test()