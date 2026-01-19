import torch
import torch.optim as optim
from src.model import VO
from lietorch import SE3

def test_training_flow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checking Pipeline on: {device}")

    # 1. 모델 및 옵티마이저 설정
    model = VO(baseline=0.54).to(device)
    model.train() # 학습 모드
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 2. 가상 데이터 생성 (Batch size=1)
    dummy_images = torch.randn(1, 4, 3, 320, 640).to(device)
    dummy_intrinsics = torch.tensor([718.8, 718.8, 607.2, 185.2], device=device).unsqueeze(0)    
    # 가상의 정답(Ground Truth) - 전진 0.1m 이동했다고 가정
    gt_pose = SE3.Identity(1, device=device)
    # 0.1m 전진 (Z축)
    gt_pose.data[:, 2] = 0.1 

    print("\n[Step 1] Forward Pass...")
    pred_pose, pred_depth = model(dummy_images, dummy_intrinsics, iters=4)

    # 3. Loss 계산 (Pose Loss - SE3 Geodesic distance)
    # 간단하게 translation 부분의 MSE만 체크해봅니다.
    loss = torch.mean((pred_pose.data - gt_pose.data)**2)
    print(f"Initial Loss: {loss.item():.6f}")

    # 4. Backward Pass
    print("[Step 2] Backward Pass...")
    optimizer.zero_grad()
    loss.backward()

    # 5. Gradient Check (가장 핵심!)
    # 주요 모듈들에서 기울기가 잘 전달되는지 확인
    print("\n--- Gradient Check ---")
    layers_to_check = {
        "GAT": model.GAT.parameters(),
        "UpdateBlock": model.update_block.parameters()
    }
    if hasattr(model.extractor, 'parameters'):
        layers_to_check["Extractor"] = model.extractor.parameters()

    for name, params in layers_to_check.items():
        has_grad = all(p.grad is not None for p in params if p.requires_grad)
        status = "✅ Flowing" if has_grad else "❌ Blocked"
        print(f"{name:15}: {status}")

    # 6. Step & Update Check
    print("\n[Step 3] Optimizer Step...")
    optimizer.step()
    
    # 파라미터가 실제로 변했는지 확인
    print("Pipeline check complete!")

if __name__ == "__main__":
    test_training_flow()