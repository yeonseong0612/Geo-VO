import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from lietorch import SE3

# 설정 및 모델 모듈 임포트
from CFG.vo_cfg import vo_cfg
from src.model import VO
from src.loader import DataFactory
from src.loss import total_loss

def setup():
    """DDP 환경 초기화"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup():
    """DDP 종료"""
    dist.destroy_process_group()

def train():
    # 1. DDP 설정
    local_rank = setup()
    device = torch.device("cuda", local_rank)
    
    # 체크포인트 저장 경로는 메인 프로세스(0번)에서만 생성
    if local_rank == 0:
        os.makedirs(vo_cfg.logdir, exist_ok=True)
        print(f"🚀 GPU {dist.get_world_size()}개에서 DDP 병렬 학습을 시작합니다.")
        print(f"📂 체크포인트 저장 경로: {vo_cfg.logdir}")

    # 2. 모델 초기화 및 DDP 적용
    model = VO(baseline=0.54).to(device)
    model = DDP(
        model, 
        device_ids=[local_rank], 
        output_device=local_rank,
        find_unused_parameters=True  
    )
    # 3. 데이터셋 및 로더 설정 (DistributedSampler 필수)
    train_set = DataFactory(vo_cfg, mode='train')
    sampler = DistributedSampler(train_set, shuffle=True)
    
    train_loader = DataLoader(
        train_set, 
        batch_size=vo_cfg.batchsize,
        sampler=sampler,
        num_workers=vo_cfg.num_cpu,
        pin_memory=False, 
        drop_last=True,
        prefetch_factor=4    
    )

    # 4. 옵티마이저 및 스케줄러
    optimizer = torch.optim.AdamW(model.parameters(), lr=vo_cfg.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=vo_cfg.MultiStepLR_milstone, 
        gamma=vo_cfg.MultiStepLR_gamma
    )

    # 5. 메인 학습 루프
    for epoch in range(vo_cfg.maxepoch):
        model.train()
        sampler.set_epoch(epoch)  # 매 에폭마다 데이터를 다르게 셔플링
        epoch_loss = 0.0
        
        # tqdm은 0번 GPU에서만 출력
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{vo_cfg.maxepoch}]") if local_rank == 0 else train_loader
        
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # 데이터 로드 및 GPU 전송
            images = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            gt_poses = SE3(batch['rel_pose'].to(device))

            # Forward Pass (이제 outputs는 해당 GPU의 독립적인 결과물입니다)
            outputs = model(images, intrinsics, iters=8)

            # [핵심] 이제 복잡한 gather_and_verify 없이 모델 출력을 그대로 사용합니다.
            # DDP가 내부적으로 Gradient를 합쳐주기 때문에 에러가 발생하지 않습니다.
            loss, l_pose, l_weight = total_loss(outputs, gt_poses, vo_cfg)            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            # 메인 프로세스에서만 상태 출력
            if local_rank == 0 and i % 5 == 0:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.6f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.6e}"
                })

        scheduler.step()
        
        # 6. 모델 저장 (0번 GPU에서만 수행)
        if local_rank == 0:
            avg_loss = epoch_loss / len(train_loader)
            checkpoint_path = os.path.join(vo_cfg.logdir, f"checkpoint_epoch_{epoch}.pth")
            
            # DDP 모델에서 원래 가중치를 저장하기 위해 .module 접근
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"✅ Epoch {epoch} 완료 | 평균 Loss: {avg_loss:.6f}")

    cleanup()
    if local_rank == 0:
        print("🏁 모든 학습이 완료되었습니다.")

if __name__ == "__main__":
    train()