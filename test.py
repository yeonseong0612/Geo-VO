import torch
import torch.optim as optim
from src.model import VO
from src.loader import DataFactory  # 방금 만든 클래스
from src.loss import total_loss   # 방금 만든 로스 함수
from lietorch import SE3

def test_real_data_flow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Checking Pipeline on: {device}")

    # 1. Config 설정 (사용자님의 기존 cfg 객체를 사용하세요)
    # 여기서는 예시로 클래스 내부의 필요한 값만 가정합니다.
    class Config:
        proj_home = "./"
        odometry_home = "/home/yskim/projects/vo-labs/data/kitti_odometry/" # 실제 경로로 수정
        traintxt = "train.txt"
        trainsequencelist = ["00"]
        color_subdir = "datasets/sequences/"
        poses_subdir = "poses/"
        calib_subdir = "datasets/sequences/"
        
    cfg = Config()

    # 2. 모델 및 데이터 로더 설정
    model = VO(baseline=0.54).to(device)
    dataset = DataFactory(cfg, mode='train')
    
    # 데이터셋에서 샘플 하나 가져오기
    sample = dataset[0] 
    
    # 텐서들 장치 이동 및 배치 차원 추가
    images = sample['images'].unsqueeze(0).to(device)      # [1, 4, 3, H, W]
    gt_pose = sample['rel_pose'].to(device)               # SE3 object (batch 1)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device) # [1, 4]

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"\n[Step 1] Forward Pass with Real Image (Seq: {sample['seq']}, Frame: {sample['imgnum']})...")
    
    # 모델의 리턴값이 (pose, depth, poses_h, weights_h, errors_h)라고 가정
    pred_pose, pred_depth, poses_h, weights_h, errors_h = model(images, intrinsics, iters=4)

    # 3. 실제 로스 계산
    loss, loss_p, loss_w = total_loss((poses_h, weights_h, errors_h), gt_pose, gamma=0.8)
    print(f"Initial Loss: {loss.item():.6f} (Pose: {loss_p.item():.6f}, Weight: {loss_w.item():.6f})")

    # 4. Backward Pass
    print("[Step 2] Backward Pass...")
    optimizer.zero_grad()
    loss.backward()

    # 5. Gradient Check
    print("\n--- Gradient Check ---")
    layers_to_check = {
        "GAT": model.GAT.parameters(),
        "UpdateBlock": model.update_block.parameters(),
        "Extractor": model.extractor.parameters()
    }

    for name, params in layers_to_check.items():
        # 기울기가 하나라도 None이 아니면 흐르는 것으로 간주
        has_grad = any(p.grad is not None for p in params if p.requires_grad)
        status = "✅ Flowing" if has_grad else "❌ Blocked"
        print(f"{name:15}: {status}")

    # 6. Optimizer Step
    optimizer.step()
    print("\nPipeline check complete with real data!")

if __name__ == "__main__":
    test_real_data_flow()