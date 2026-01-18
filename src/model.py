import torch

from .extractor import SuperPointExtractor
from utils.geo_utils import *
from .layer import *


class VO(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = SuperPointExtractor()
        self.DT = compute_delaunay_edges
        self.GAT = GeometricGAT(in_channels=258, out_channels=256)

    def run(self, img_tensor):
        # 1. 특징점 추출
        kpts_tensor, desc_tensor, scores = self.extractor(img_tensor)
        # kpts_tensor shape: [N, 2], desc_tensor shape: [256, N] (보통 SP 출력)
        
        # 2. 디스크립터 차원 정렬 [N, 256]
        # GAT 입력은 [점의 개수, 특징량] 이어야 하므로, 만약 [256, N]이라면 전치합니다.
        if desc_tensor.shape[0] == 256 and desc_tensor.shape[1] != 256:
            desc_tensor = desc_tensor.T
            
        # 3. 좌표 정규화 [N, 2]
        h, w = img_tensor.shape[2:]
        kpts_norm = kpts_tensor.clone()
        kpts_norm[:, 0] = kpts_norm[:, 0] / w
        kpts_norm[:, 1] = kpts_norm[:, 1] / h
        
        # 4. 특징 결합 (Concat) -> [N, 258]
        # 이제 desc_tensor[N, 256]와 kpts_norm[N, 2]의 행(N)이 일치하므로 안전하게 합쳐집니다.
        node_features = torch.cat([desc_tensor, kpts_norm], dim=1)
        
        # 5. DT 전용 Numpy 변환 (좌표값만)
        kpts_np = kpts_tensor.detach().cpu().numpy()
        edges_np = self.DT(kpts_np) 
        
        # 6. GAT 전용 Tensor 변환 [2, E]
        if edges_np.shape[0] != 2: # 만약 [E, 2]라면 [2, E]로 변경
            edges_np = edges_np.T
        edges = torch.from_numpy(edges_np).to(img_tensor.device).long()
        
        # 7. edge_attr 계산 (기하학적 거리)
        src_pts = kpts_tensor[edges[0]]
        dst_pts = kpts_tensor[edges[1]]
        edge_attr = torch.norm(src_pts - dst_pts, dim=1, keepdim=True)

        # 8. GAT 통과
        # 주의: self.GAT의 in_channels는 반드시 258이어야 합니다.
        refined_desc, attn = self.GAT(node_features, edges, edge_attr)

        return kpts_tensor, refined_desc, attn