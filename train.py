import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm  # 진행률 표시줄을 위해 추가 권장

# 설정 및 모델 모듈 임포트
from CFG.vo_cfg import vo_cfg
from src.model import VO
from src.loader import DataFactory
from src.loss import total_loss

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. WandB 초기화 (실험 추적용)
    wandb.init(project=vo_cfg.model, config=vo_cfg)
    
    # 2. 모델 및 데이터 로더 설정
    model = VO(baseline=0.54).to(device)
    train_set = DataFactory(vo_cfg, mode='train')
    train_loader = DataLoader(
        train_set, 
        batch_size=vo_cfg.batchsize, 
        shuffle=True, 
        num_workers=vo_cfg.num_cpu
    )

    # 3. 옵티마이저 및 학습률 스케줄러
    optimizer = torch.optim.AdamW(model.parameters(), lr=vo_cfg.learing_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=vo_cfg.MultiStepLR_milstone, 
        gamma=vo_cfg.MultiStepLR_gamma
    )
    
    # 체크포인트 저장 경로 생성
    os.makedirs(vo_cfg.logdir, exist_ok=True)

    print(f"🚀 {vo_cfg.model} 학습 시작 (Device: {device})")

    # 4. 메인 학습 루프
    for epoch in range(vo_cfg.maxepoch):
        model.train()
        epoch_loss = 0.0
        
        # tqdm을 사용하면 터미널에서 진행 상황을 보기 편합니다.
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{vo_cfg.maxepoch}]")
        
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # 데이터 가져오기 (DataFactory의 출력 키값에 따라 수정 필요할 수 있음)
            images = batch['images'].to(device)       # [B, 4, 3, H, W]
            intrinsics = batch['intrinsics'].to(device) # [B, 4]
            gt_poses = batch['poses'].to(device)         # [B, 4, 4]

            # Forward Pass
            # iters 값은 모델 설계에 따라 조정하세요.
            pred_poses, pred_depths = model(images, intrinsics, iters=8)
            
            # 5. Loss 계산 (src.loss.total_loss 사용)
            # 포즈 오차와 필요시 깊이 오차 등을 종합해서 계산
            loss = total_loss(pred_poses, gt_poses) 
            
            # Backward Pass & Optimization
            loss.backward()
            
            # 안정성을 위한 Gradient Clipping (권장)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 통계 기록
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})
            
            # 실시간 WandB 로그
            if i % 10 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                })

        # 한 에폭 종료 후 스케줄러 업데이트
        scheduler.step()
        
        # 6. 모델 저장 (체크포인트)
        avg_loss = epoch_loss / len(train_loader)
        print(f"✅ Epoch {epoch} 완료 | 평균 Loss: {avg_loss:.6f}")
        
        checkpoint_path = os.path.join(vo_cfg.logdir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        wandb.log({"epoch_avg_loss": avg_loss, "epoch": epoch})

    print("🏁 모든 학습이 완료되었습니다.")
    wandb.finish()

if __name__ == "__main__":
    train()