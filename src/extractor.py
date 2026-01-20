import torch
import numpy as np
from thirdparty.superpoint.superpoint import SuperPoint

class SuperPointExtractor:
    def __init__(self, weights="/home/jnu-ie/kys/Geo-VO/models/superpoint_v6_from_tf.pth",
                 max_keypoints=1000, device="cpu"):
        self.model = SuperPoint(max_num_keypoints=max_keypoints).to(device)
        state = torch.load(weights, map_location=device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image_tensor):
        image_tensor = image_tensor.to(self.device)

        if image_tensor.shape[1] == 3:
            image_tensor = image_tensor.mean(dim=1, keepdim=True)

        out = self.model({"image": image_tensor})
        
        kpts = out["keypoints"]          
        desc = out["descriptors"]         
        fixed_n = 1000
        new_kpts = []
        new_desc = []

        for k, d in zip(kpts, desc):
            # k의 크기가 [N, 2], d의 크기가 [N, C]라고 가정
            num_k = k.shape[0]
            if num_k >= fixed_n:
                new_kpts.append(k[:fixed_n])
                new_desc.append(d[:fixed_n])
            else:
                # 모자라면 마지막 값을 복사하거나 제로 패딩 (모델 설계에 따라 다름)
                padding_k = torch.zeros((fixed_n - num_k, 2), device=k.device)
                padding_d = torch.zeros((fixed_n - num_k, d.shape[1]), device=d.device)
                new_kpts.append(torch.cat([k, padding_k], dim=0))
                new_desc.append(torch.cat([d, padding_d], dim=0))

        kpts = torch.stack(new_kpts)
        desc = torch.stack(new_desc)

        desc = desc.transpose(1, 2)

        return kpts, desc