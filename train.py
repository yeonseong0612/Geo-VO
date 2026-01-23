import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
from datetime import datetime

from CFG.vo_cfg import vo_cfg as cfg
from src.model import VO
from src.loader import DataFactory, vo_collate_fn
from src.loss import total_loss

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
        num_workers=cfg.num_cpu,
        sampler=sampler, 
        collate_fn=vo_collate_fn,
        pin_memory=True
    )

    model = VO(cfg).to(device)
    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    raw_model = model.module if is_ddp else model
    
    params = [
        {'params': [p for n, p in raw_model.named_parameters() if 'log_lmbda' not in n], 'lr': cfg.learning_rate},
        {'params': [raw_model.log_lmbda], 'lr': 1e-3}
    ]
    
    optimizer = optim.AdamW(params, cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.MultiStepLR_milstone, gamma=cfg.MultiStepLR_gamma
    )

    writer = None
    log_file = None
    if rank == 0:
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        
        log_path = os.path.join(cfg.logdir, f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt")
        log_file = open(log_path, "w")
        log_file.write(f"Training Start: {datetime.now()}\n")
        log_file.write(f"Config: Epochs={cfg.maxepoch}, BatchSize={cfg.batchsize}, LR={cfg.learning_rate}\n")
        log_file.write("-" * 100 + "\n")
        log_file.flush()

        writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, 'tensorboard'))
        print(f"==> 학습 시작: Log 저장위치={log_path}")

    for epoch in range(cfg.maxepoch):
        if is_ddp:
            sampler.set_epoch(epoch)
        
        model.train()
        # --- 지표 누적을 위한 변수 초기화 ---
        epoch_loss = 0.0
        epoch_t_err = 0.0
        epoch_r_err = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            gt_pose = batch['rel_pose'].to(device)
            # noise = torch.randn_like(gt_pose) * 0.01 
            # gt_guide_pose = gt_pose + noise

            outputs = model(batch, iters=12, gt_guide=None)

            loss, t_err, r_err, l_w = total_loss(outputs, gt_pose)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 현재 람다 값 모니터링
            curr_lmbda = torch.exp(raw_model.log_lmbda).item()

            # --- 지표 누적 및 평균 계산 ---
            epoch_loss += loss.item()
            epoch_t_err += t_err
            epoch_r_err += r_err
            
            avg_loss = epoch_loss / (i + 1)
            avg_t_err = epoch_t_err / (i + 1)
            avg_r_err = epoch_r_err / (i + 1)

            if rank == 0:
                # 진행 바에는 실시간 평균값을 표시
                pbar.set_postfix({
                    "AvgL": f"{avg_loss:.3f}", 
                    "AvgT": f"{avg_t_err:.3f}", 
                    "AvgR": f"{avg_r_err:.3f}",
                    "Lm": f"{curr_lmbda:.1e}"
                })

                global_step = epoch * len(loader) + i
                writer.add_scalar('Batch/Loss', loss.item(), global_step)
                writer.add_scalar('Batch/Trans_Err', t_err, global_step)
                writer.add_scalar('Batch/Rot_Err', r_err, global_step)
                writer.add_scalar('Batch/Weight_Loss', l_w, global_step)
                writer.add_scalar('Batch/Lambda', curr_lmbda, global_step)

        scheduler.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            # 로그 파일에는 에포크 전체 평균값을 기록
            log_str = (f"[Epoch {epoch}] AvgL: {avg_loss:.5f} | AvgT: {avg_t_err:.5f} | AvgR: {avg_r_err:.5f} | "
                       f"WLoss: {l_w:.7f} | Lm: {curr_lmbda:.2e} | LR: {current_lr:.7f}\n")
            print(f"\n{log_str}")
            
            log_file.write(log_str)
            log_file.flush()

            writer.add_scalar('Epoch/Avg_Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/Avg_Trans_Err', avg_t_err, epoch)
            writer.add_scalar('Epoch/Avg_Rot_Err', avg_r_err, epoch)
            writer.add_scalar('Epoch/Lambda', curr_lmbda, epoch)

            save_path = os.path.join(cfg.logdir, f"vo_model_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                't_err': avg_t_err,
                'r_err': avg_r_err,
                'lmbda': curr_lmbda
            }, save_path)

    if rank == 0:
        log_file.write(f"Training Finished: {datetime.now()}\n")
        log_file.close()
        writer.close()
    if is_ddp:
        cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        train(0, 1, cfg)