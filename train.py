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

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.MultiStepLR_milstone, gamma=cfg.MultiStepLR_gamma
    )

    writer = None
    if rank == 0:
        if not os.path.exists(cfg.logdir):
            os.makedirs(cfg.logdir)
        writer = SummaryWriter(log_dir=os.path.join(cfg.logdir, 'tensorboard'))
        print(f"==> 학습 시작: Epochs={cfg.maxepoch}, BatchSize={cfg.batchsize}")

    for epoch in range(cfg.maxepoch):
        if is_ddp:
            sampler.set_epoch(epoch)
        
        model.train()
        
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        
        for i, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            gt_pose = batch['rel_pose'].to(device)
            outputs = model(batch, iters=8) 
            
            loss, l_p, l_w = total_loss(outputs, gt_pose)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            avg_loss = epoch_loss / (i + 1)

            if rank == 0:
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "Avg": f"{avg_loss:.4f}", 
                    "Pose": f"{l_p:.4f}"
                })

                global_step = epoch * len(loader) + i
                writer.add_scalar('Batch/Loss', loss.item(), global_step)
                writer.add_scalar('Batch/Pose_Loss', l_p, global_step)
                writer.add_scalar('Batch/Weight_Loss', l_w, global_step)

        scheduler.step()

        if rank == 0:
            writer.add_scalar('Epoch/Avg_Loss', avg_loss, epoch)
            writer.add_scalar('Epoch/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            save_path = os.path.join(cfg.logdir, f"vo_model_{epoch}.pth")
            state_dict = model.module.state_dict() if is_ddp else model.state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
            print(f"\n==> Epoch {epoch} Saved (Avg Loss: {avg_loss:.4f})")

    if rank == 0:
        writer.close()
    if is_ddp:
        cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        train(0, 1, cfg)