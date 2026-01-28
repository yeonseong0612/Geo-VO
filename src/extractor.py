import torch
import torch.nn as nn
from thirdparty.superpoint.superpoint import SuperPoint
class SuperPointExtractor:
    def __init__(self, weights="/home/yskim/projects/Geo-VO/superpoint_v6_from_tf.pth",
                 max_keypoints=1000, device="cuda"):
        # 1. 모델 로드 시 보안 및 경고 해결
        self.model = SuperPoint(max_num_keypoints=max_keypoints).to(device)
        try:
            state = torch.load(weights, map_location=device, weights_only=True)
            self.model.load_state_dict(state, strict=False)
        except Exception as e:
            print(f"가중치 로드 실패: {e}. 경로를 확인하세요.")
            
        self.model.eval()
        self.device = device
        self.fixed_n = 800  

    @torch.no_grad()
    def __call__(self, image_tensor):
        # 2. 확실한 채널 전처리
        image_tensor = image_tensor.to(self.device).float()
        
        # [B, C, H, W] 구조인지 확인 후 흑백 변환
        if image_tensor.ndim == 4 and image_tensor.shape[1] == 3:
            image_tensor = image_tensor.mean(dim=1, keepdim=True)
        elif image_tensor.ndim == 3: # 배치가 없는 경우 대비
            image_tensor = image_tensor.unsqueeze(0)
            if image_tensor.shape[1] == 3:
                image_tensor = image_tensor.mean(dim=1, keepdim=True)

        # 3. 모델 추론
        out = self.model({"image": image_tensor})
        
        # SuperPoint의 결과는 리스트 형태이므로 배치 단위로 처리
        kpts = out["keypoints"]   
        desc = out["descriptors"] 
        
        final_kpts = []
        final_desc = []

        for k, d in zip(kpts, desc):
            # 디스크립터 차원 정렬 [N, 256] 확인
            if d.shape[0] == 256 and d.shape[1] != 256:
                d = d.transpose(0, 1)

            num_k = k.shape[0]
            
            if num_k >= self.fixed_n:
                # 상위 fixed_n개만 선택 (SuperPoint는 보통 score 순으로 정렬되어 나옴)
                final_kpts.append(k[:self.fixed_n])
                final_desc.append(d[:self.fixed_n])
            elif num_k > 0:
                # 부족한 부분을 0으로 채우는 방식이 모델 성능에 더 유리함
                diff = self.fixed_n - num_k
                k_padded = torch.cat([k, torch.zeros((diff, 2), device=k.device)], dim=0)
                d_padded = torch.cat([d, torch.zeros((diff, 256), device=d.device)], dim=0)
                
                final_kpts.append(k_padded)
                final_desc.append(d_padded)
            else:
                # 특징점이 하나도 없는 경우
                final_kpts.append(torch.zeros((self.fixed_n, 2), device=self.device))
                final_desc.append(torch.zeros((self.fixed_n, 256), device=self.device))

        return torch.stack(final_kpts), torch.stack(final_desc)