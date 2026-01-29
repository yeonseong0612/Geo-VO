import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm 
from datetime import datetime

from CFG.vo_cfg import vo_cfg as cfg
from src.model import VO
from src.loader import DataFactory, vo_collate_fn
from src.loss import total_loss

# [추가] 역전파 이상 탐지 활성화 - NaN이 발생한 연산 지점을 정확히 짚어줍니다.
torch.autograd.set_detect_anomaly(True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, cfg):
    is_ddp = world_size > 1
    if is_ddp:
        setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    dataset = DataFactory(cfg, mode='train')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batchsize, 
        shuffle=(sampler is None),
        num_workers=0,
        sampler=sampler, 
        collate_fn=vo_collate_fn,
        pin_memory=True
    )

    model = VO(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    
    raw_model = model.module if is_ddp else model
    
    # log_lmbda는 상수로 고정되었으므로 제외된 params 리스트 사용
    params = [
        {'params': [p for n, p in raw_model.named_parameters() if 'log_lmbda' not in n], 'lr': cfg.learning_rate}
    ]

    optimizer = optim.AdamW(params, cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.MultiStepLR_milstone, gamma=cfg.MultiStepLR_gamma
    )

    log_file = None
    if rank == 0:
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        
        log_path = os.path.join(cfg.logdir, f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt")
        log_file = open(log_path, "w")
        log_file.write(f"Training Start: {datetime.now()}\n")
        log_file.write(f"Anomaly Detection: ON\n") # 이상 탐지 모드 기록
        log_file.write(f"Config: Epochs={cfg.maxepoch}, BatchSize={cfg.batchsize}, LR={cfg.learning_rate}\n")
        log_file.write("-" * 100 + "\n")
        log_file.flush()
        print(f"==> 학습 시작 (이상 탐지 모드): Log 저장위치={log_path}")

    for epoch in range(cfg.maxepoch):
        if is_ddp:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        epoch_t_err = 0.0
        epoch_r_err = 0.0
        epoch_l_w = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            # Forward 연산
            outputs = model(batch, iters=8) 
            
            with torch.no_grad():
                init_R = outputs['pose_matrices'][0]
                det_val = torch.det(init_R[:, :3, :3])
                # Det(R)이 NaN이면 이미 Forward에서 터진 것
                if torch.isnan(det_val).any():
                    print(f"\n[Rank {rank}] !!! Forward NaN Detected in Det(R) at Batch {i} !!!")

            loss, t_err, r_err, l_w = total_loss(outputs, batch)
            
            if torch.isnan(loss):
                print(f"\n[Rank {rank}] Warning: NaN loss detected at batch {i}. Skipping.")
                continue

            # [핵심] Backward 연산 시 이상 탐지 작동
            try:
                loss.backward()
            except RuntimeError as e:
                print("\n" + "!"*60)
                print(f"🚨 [Rank {rank}] Backward NaN Detected at Batch {i}!")
                print(f"Error Details: {e}")
                print("!"*60)
                # 로그에 에러 기록 후 종료
                if rank == 0:
                    log_file.write(f"ERROR at Batch {i}: {str(e)}\n")
                    log_file.close()
                return # 학습 중단

            # 그래디언트 클리핑 (더 보수적으로 0.1 설정)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            # 지표 업데이트
            epoch_loss += loss.item()
            epoch_t_err += t_err
            epoch_r_err += r_err
            epoch_l_w += l_w.item() if torch.is_tensor(l_w) else l_w
            
            avg_loss = epoch_loss / (i + 1)
            avg_t_err = epoch_t_err / (i + 1)
            avg_r_err = epoch_r_err / (i + 1)

            curr_lmbda = torch.exp(raw_model.log_lmbda).item()

            if rank == 0:
                pbar.set_postfix({
                    "L": f"{avg_loss:.3f}", 
                    "T": f"{avg_t_err:.3f}m", 
                    "R": f"{avg_r_err:.3f}rad",
                    "Lm": f"{curr_lmbda:.1e}"
                })

        scheduler.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_str = (f"[Epoch {epoch}] AvgL: {avg_loss:.5f} | AvgT: {avg_t_err:.5f} | AvgR: {avg_r_err:.5f} | "
                       f"Lm: {curr_lmbda:.2e} | LR: {current_lr:.7f}\n")
            print(f"\n{log_str}")
            log_file.write(log_str)
            log_file.flush()

            save_path = os.path.join(cfg.logdir, f"vo_model_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'lmbda': curr_lmbda
            }, save_path)

    if rank == 0:
        log_file.write(f"Training Finished: {datetime.now()}\n")
        log_file.close()
    if is_ddp:
        cleanup()

if __name__ == "__main__":
    train(0, 1, cfg)