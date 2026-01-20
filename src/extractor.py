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
        
        fixed_n = 800 
        final_kpts = []
        final_desc = []

        for k, d in zip(kpts, desc):
            num_k = k.shape[0]
            keep_n = min(num_k, fixed_n)
            
            final_kpts.append(k[:keep_n])
            final_desc.append(d[:keep_n])


        return final_kpts, final_desc