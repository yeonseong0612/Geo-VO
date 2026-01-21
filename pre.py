import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T

from src.model import VO
from CFG.vo_cfg import vo_cfg  

class LastFrameDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.samples = []
        
        # KITTI 표준 해상도 고정 (Dataloader stack 에러 방지)
        self.target_h, self.target_w = 376, 1241 
        self.transform = T.Compose([
            T.Resize((self.target_h, self.target_w)), 
            T.ToTensor()
        ])
        
        # 시퀀스 리스트 할당
        target_seqs = getattr(cfg, 'trainsequencelist', getattr(cfg, 'sequences', []))
        self.base_path = "/home/jnu-ie/Dataset/kitti_odometry/data_odometry_color/dataset/sequences/"

        print(f"🔍 탐색할 시퀀스 리스트: {target_seqs}")

        for seq in target_seqs:
            img_dir = os.path.join(self.base_path, seq, "image_2")
            if not os.path.exists(img_dir):
                continue
            
            fnames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
            if fnames:
                last_f = fnames[-1]
                img_num = int(last_f.split('.')[0])
                self.samples.append((seq, img_num))
                print(f"✅ 발견: Sequence {seq}의 마지막 프레임은 {img_num}번 입니다.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, img_num = self.samples[idx]
        path_L = os.path.join(self.base_path, seq, "image_2", f"{str(img_num).zfill(6)}.png")
        path_R = os.path.join(self.base_path, seq, "image_3", f"{str(img_num).zfill(6)}.png")
        
        img_L = self.transform(Image.open(path_L).convert('RGB'))
        img_R = self.transform(Image.open(path_R).convert('RGB'))
        
        return {
            'images': torch.stack([img_L, img_R], dim=0), 
            'seq': seq, 
            'imgnum': img_num
        }

@torch.no_grad()
def export_last_only(model, dataloader, save_dir):
    model.eval()
    # 1. 사용할 장치를 명시적으로 고정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    from scipy.spatial import Delaunay

    print(f"🎯 누락된 마지막 프레임({len(dataloader.dataset)}개) 전처리 시작...")

    for batch in tqdm(dataloader):
        images = batch['images'].to(device)
        seqs = batch['seq']
        imgnums = batch['imgnum']
        
        # 이미지 크기 추출
        h, w = images.shape[-2:]
        # size_vec을 연산 시점에 device에 딱 맞게 생성
        size_vec = torch.tensor([w, h], dtype=torch.float32, device=device).view(1, 2)

        for side_idx, side_name in zip([0, 1], ['image_2', 'image_3']):
            # extractor 결과 (list of tensors)
            kpts_list, desc_list = model.extractor(images[:, side_idx])
            
            for b in range(len(kpts_list)):
                # [수정 핵심] 개별 텐서를 한 번 더 명시적으로 device로 이동
                k = kpts_list[b].to(device)
                d = desc_list[b].to(device)
                
                # GPU 상에서 정규화 연산 수행
                k_norm = k / size_vec 
                    
                if d.shape[0] == 256: 
                    d = d.transpose(0, 1)
                
                # CPU 기반 후처리(Delaunay, Save)를 위해 넘파이 변환
                node_feat = torch.cat([d, k_norm], dim=-1).cpu().numpy()
                k_np = k.cpu().numpy()

                # Delaunay & Edges 생성
                if len(k_np) < 3:
                    edges_np = np.zeros((2, 0), dtype=np.int32)
                else:
                    tri = Delaunay(k_np)
                    edges = np.concatenate([
                        tri.simplices[:, [0, 1]], 
                        tri.simplices[:, [1, 2]], 
                        tri.simplices[:, [2, 0]]
                    ], axis=0)
                    edges_np = np.unique(np.sort(edges, axis=1), axis=0).T

                # Edge Attributes (Euclidean Distance)
                if edges_np.shape[1] > 0:
                    edge_attr = np.linalg.norm(k_np[edges_np[0]] - k_np[edges_np[1]], axis=1, keepdims=True)
                else:
                    edge_attr = np.zeros((0, 1), dtype=np.float32)

                # 시퀀스별 폴더 구조 생성 및 저장
                full_path = os.path.join(save_dir, seqs[b], side_name, f"{str(imgnums[b].item()).zfill(6)}.npz")
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                np.savez_compressed(
                    full_path, 
                    node_features=node_feat.astype(np.float16), 
                    edges=edges_np.astype(np.int32), 
                    edge_attr=edge_attr.astype(np.float16), 
                    kpts=k_np.astype(np.float32)
                )

if __name__ == "__main__":
    SAVE_PATH = "/home/jnu-ie/kys/Geo-VO/geovo_precomputed"
    
    # 모델 로드
    model = VO(vo_cfg).cuda()
    print("✅ 모델 및 설정 로드 완료")

    dataset = LastFrameDataset(vo_cfg)
    # 이미지 사이즈가 달라 stack 에러가 날 수 있으므로 num_workers=0으로 테스트하거나
    # 이미 Dataset 단계에서 Resize를 넣었으므로 그대로 사용 가능합니다.
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    
    if len(dataset) > 0:
        export_last_only(model, dataloader, SAVE_PATH)
        print("✨ 모든 시퀀스의 마지막 프레임 복구 완료!")
    else:
        print("💡 처리할 데이터가 없습니다.")