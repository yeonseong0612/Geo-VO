import torch

from .extractor import SuperPointExtractor
from utils.geo_utils import *
from .layer import *
from utils.DBA_utils import *


class VO(nn.Module):
    def __init__(self, baseline=0.54):
        super().__init__()
        self.extractor = SuperPointExtractor()
        self.DT = compute_delaunay_edges
        self.GAT = GeometricGAT(in_channels=258, out_channels=256)
        self.corr_block = CorrBlock()
        self.cyclic_module = CyclicErrorModule(baseline)
        self.update_block = UpdateBlock(hidden_dim=256)
        self.init_depth_net = nn.Linear(256, 1)
        self.DBA = DBASolver()
        self.DBA_Updater = PoseDepthUpdater()
        self.lmbda = nn.Parameter(torch.tensor(1e-4))

    def forward(self, images_tensor, intrinsics, iters=8):
        B = images_tensor.shape[0] 
        device = images_tensor.device
        intrinsics = intrinsics.to(device)
        
        results = {}
        img_names = ['Lt', 'Rt', 'Lt1', 'Rt1']

        for i, name in enumerate(img_names):
            # run_single_img가 이제 [B, N, D]를 반환함
            kpts, desc, attn, edges = self.run_single_img(images_tensor[:, i])
            results[name] = {'kpts': kpts, 'desc': desc}
            
        f_Lt = results['Lt']['desc']      # [B, N, 256]
        kpts_Lt = results['Lt']['kpts']  # [B, N, 2]
        N = kpts_Lt.shape[1]             # index 1이 N (특징점 개수)

        curr_pose = SE3.Identity(B, device=device)
        curr_depth = torch.ones((B, N, 1), device=device) * 5.0
        h = torch.zeros((B, N, 256), device=device)     

        # 3. 업데이트 루프
        for i in range(iters):
            # corr_block도 [B, N, D] 입력을 받도록 설계되어 있어야 함
            c_temp = self.corr_block(f_Lt, results['Lt1']['desc'])
            c_stereo = self.corr_block(f_Lt, results['Rt']['desc'])
            
            # cyclic_module은 이제 [B, N, 1] depth와 [B] pose를 받음
            e_proj = self.cyclic_module(kpts_Lt, curr_depth, curr_pose, intrinsics)
            e_proj_in = e_proj.unsqueeze(-1) # [B, N, 2] -> [B, N, 2, 1]?

            h, r, w, a_p, a_d = self.update_block(
                h, c_temp, c_stereo, e_proj, f_Lt 
            )
            J_p, J_d = compute_projection_jacobian(kpts_Lt, curr_depth, intrinsics)
            delta_pose_dba, delta_depth_dba = self.DBA(r, w, J_p, J_d, self.lmbda)
            curr_pose, curr_depth = self.DBA_Updater(curr_pose, curr_depth, delta_pose_dba, delta_depth_dba, a_p, a_d)
            
        return curr_pose, curr_depth.view(B, N, 1)

            
    def run_single_img(self, img_tensor):
        device = img_tensor.device 
        B = img_tensor.shape[0] # 배치 사이즈 확인
    
        # 1. 특징점 추출 (배치 유지 [B, N, D])
        kpts_tensor, desc_tensor = self.extractor(img_tensor)
        kpts_tensor = kpts_tensor.to(device)
        desc_tensor = desc_tensor.to(device)

        # 2. 디스크립터 전치 [B, N, 256]
        if desc_tensor.shape[1] == 256: 
            desc_tensor = desc_tensor.transpose(1, 2)
                
        # 3. 좌표 정규화
        h, w = img_tensor.shape[-2:]
        size_tensor = torch.tensor([w, h], device=device, dtype=torch.float32).view(1, 1, 2)
        kpts_norm = kpts_tensor / size_tensor 
        
        # 4. 특징 결합 [B, N, 258]
        node_features = torch.cat([desc_tensor, kpts_norm], dim=-1)
        B, N, D = node_features.shape

        # 5. DT 및 Edges 계산 
        # (주의: DT가 배치 단위 처리를 지원하지 않는다면 루프를 돌아야 하지만, 일단 B=1 기준으로 작성)
        kpts_np = kpts_tensor[0].detach().cpu().numpy() # 첫 번째 배치 샘플 사용
        edges_np = self.DT(kpts_np) 
        if edges_np.shape[0] != 2:
            edges_np = edges_np.T
            
        edges = torch.from_numpy(edges_np).to(device).long()
        
        # edge_attr 계산 (배치 0번 기준)
        src_pts = kpts_tensor[0][edges[0]]
        dst_pts = kpts_tensor[0][edges[1]]
        edge_attr = torch.norm(src_pts - dst_pts, dim=1, keepdim=True)

        # 7. GAT 통과: [B, N, D] -> [B*N, D]로 펼쳐서 GAT 제약 조건 해결
        node_features_2d = node_features.view(-1, D)
        
        # GAT 연산
        refined_desc_2d, attn = self.GAT(node_features_2d, edges, edge_attr)
        
        # 다시 배치 차원 복구: [B*N, D_out] -> [B, N, D_out]
        refined_desc = refined_desc_2d.view(B, N, -1)

        return kpts_tensor, refined_desc, attn, edges