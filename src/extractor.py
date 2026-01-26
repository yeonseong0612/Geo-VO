import torch
import torch.nn as nn
from thirdparty.superpoint.superpoint import SuperPoint

class SuperPointExtractor:
    def __init__(self, weights="/home/yskim/projects/Geo-VO/superpoint_v6_from_tf.pth",
                 max_keypoints=1000, device="cuda"):
        self.model = SuperPoint(max_num_keypoints=max_keypoints).to(device)
        state = torch.load(weights, map_location=device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.device = device
        self.fixed_n = 800  

    @torch.no_grad()
    def __call__(self, image_tensor):
        image_tensor = image_tensor.to(self.device)

        if image_tensor.shape[1] == 3:
            image_tensor = image_tensor.mean(dim=1, keepdim=True)

        out = self.model({"image": image_tensor})
        
        kpts = out["keypoints"]    # List of [N, 2]
        desc = out["descriptors"]  # List of [N, 256]
        
        final_kpts = []
        final_desc = []

        for k, d in zip(kpts, desc):
            if d.shape[0] == 256 and d.shape[1] != 256:
                d = d.transpose(0, 1)

            num_k = k.shape[0]
            
            if num_k >= self.fixed_n:
                final_kpts.append(k[:self.fixed_n])
                final_desc.append(d[:self.fixed_n])
            elif num_k > 0:
                diff = self.fixed_n - num_k
                idx = torch.randint(0, num_k, (diff,), device=k.device)
                
                k_padded = torch.cat([k, k[idx]], dim=0)
                d_padded = torch.cat([d, d[idx]], dim=0)
                
                final_kpts.append(k_padded)
                final_desc.append(d_padded)
            else:
                final_kpts.append(torch.zeros((self.fixed_n, 2), device=k.device))
                final_desc.append(torch.zeros((self.fixed_n, 256), device=d.device))

        return torch.stack(final_kpts), torch.stack(final_desc)