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
    
    device = torch.device(f"cuda:{rank}")
    
    # 1. 데이터 로더 설정
    dataset = DataFactory(cfg, mode='train')
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if is_ddp else None
    loader = DataLoader(
        dataset, 
        batch_size=cfg.batchsize, 
        shuffle=(sampler is None),
        num_workers=4, # 성능을 위해 4 정도로 상향 권장
        sampler=sampler, 
        collate_fn=vo_collate_fn,
        pin_memory=True
    )

    # 2. 모델 설정
    model = VO(cfg).to(device)
    
    # [수정] 체크포인트 로드 (성공적이었던 에포크 4 불러오기)
    checkpoint_path = "./checkpoint/geovo_epoch_4.pth"
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # DDP 저장 방식에 따라 'module.' 접두사 제거가 필요할 수 있음
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)
        start_epoch = checkpoint['epoch'] + 1
        if rank == 0:
            print(f"✅ 체크포인트 로드 성공: {checkpoint_path} (에포크 {start_epoch}부터 재개)")
            
    # [핵심] log_lmbda 고정 (수치적 안정성 확보)
    model.log_lmbda.requires_grad = False
    
    if is_ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        raw_model = model.module
    else:
        raw_model = model
    
    # 3. 옵티마이저 설정
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=cfg.learning_rate * 0.2, 
        weight_decay=cfg.weight_decay
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.MultiStepLR_milstone, gamma=cfg.MultiStepLR_gamma)

    # 4. 로그 파일 설정
    log_file = None
    if rank == 0:
        if not os.path.exists(cfg.logdir): os.makedirs(cfg.logdir)
        log_path = os.path.join(cfg.logdir, f"train_log_{datetime.now().strftime('%m%d_%H%M')}.txt")
        log_file = open(log_path, "w")
        print(f"🚀 Fine-tuning 시작 | GPU 개수: {world_size} | 로그: {log_path}")

    # 학습 루프
    for epoch in range(start_epoch, cfg.maxepoch):
        if is_ddp: sampler.set_epoch(epoch)
        model.train()
        
        # [에러 해결] 각 에폭 시작 시 모니터링 변수 초기화
        avg_loss, avg_t, avg_r = 0.0, 0.0, 0.0
        epoch_loss, epoch_t, epoch_r = 0.0, 0.0, 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=(rank != 0))
        
        for i, batch in enumerate(pbar):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            optimizer.zero_grad()
            
            # 모델 추론 (iters=4 유지, 재투영 오차 포함)
            outputs = model(batch, iters=4, mode='train')
            
            # [수정] total_loss 반환값 개수 일치 (final_loss, t_err, r_err, l_weight)
            loss, t_err, r_err, l_weight = total_loss(outputs, batch)

            if torch.isnan(loss):
                print(f"⚠️ Skip NaN Loss at Epoch {epoch}, Batch {i}")
                continue

            loss.backward()

            # [핵심] Gradient Clipping: 0.94m 정체기 돌파 시 갑작스러운 폭주 방지
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            
            # 통계 업데이트
            loss_val = loss.item()
            epoch_loss += loss_val
            epoch_t += t_err
            epoch_r += r_err
            
            if rank == 0:
                # 이동 평균 계산
                avg_loss = (avg_loss * i + loss_val) / (i + 1)
                avg_t = (avg_t * i + t_err) / (i + 1)
                avg_r = (avg_r * i + r_err) / (i + 1)

                pbar.set_postfix({
                    'L(avg/cur)': f"{avg_loss:.3f}/{loss_val:.3f}",
                    'T(avg/cur)': f"{avg_t:.3f}/{t_err:.3f}m",
                    'R(avg/cur)': f"{avg_r:.4f}/{r_err:.4f}r"
                })

        scheduler.step()

        # 에포크 종료 후 저장 및 기록
        if rank == 0:
            final_avg_loss = epoch_loss / len(loader)
            final_avg_t = epoch_t / len(loader)
            final_avg_r = epoch_r / len(loader)
            
            log_str = f"[Epoch {epoch}] Avg Loss: {final_avg_loss:.4f}, Avg T: {final_avg_t:.4f}m, Avg R: {final_avg_r:.6f}rad\n"
            log_file.write(log_str)
            log_file.flush()

            checkpoint_dir = "./checkpoint/5"
            if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
            
            save_path = os.path.join(checkpoint_dir, f"geovo_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': final_avg_loss,
                'log_lmbda': raw_model.log_lmbda.data
            }, save_path)
            print(f"💾 Epoch {epoch} 모델 저장 완료: {save_path}")

    if is_ddp: cleanup()
    if log_file: log_file.close()

def main():
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(train, args=(world_size, cfg), nprocs=world_size, join=True)
    else:
        train(0, 1, cfg)

if __name__ == "__main__":
    main()