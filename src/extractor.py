import torch
import numpy as np
from thirdparty.superpoint.superpoint import SuperPoint

class SuperPointExtractor:
    def __init__(self, weights="/Users/yeonseongsmac/01_Projects/Geo-VO/models/superpoint_v6_from_tf.pth",
                 max_keypoints=1000, device="cpu"):
        self.model = SuperPoint(max_num_keypoints=max_keypoints).to(device)
        state = torch.load(weights, map_location=device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, image_tensor):
        # 1. 이미 Tensor가 들어오므로 변환 과정 생략 (또는 안전하게 검사)
        if isinstance(image_tensor, np.ndarray):
            image_tensor = torch.from_numpy(image_tensor/255.).float()[None, None]
        
        image_tensor = image_tensor.to(self.device)
        
        # 2. 모델 추론
        out = self.model({"image": image_tensor})
        
        # 3. Tensor 상태 그대로 리턴 (Batch 차원 제거)
        kpts = out["keypoints"][0]            # (N, 2) Tensor
        desc = out["descriptors"][0].T        # (N, 256) Tensor로 Transpose
        scores = out["keypoint_scores"][0]    # (N,) Tensor
        
        return kpts, desc, scores