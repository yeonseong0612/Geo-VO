import torch
import torch.nn as nn
from lietorch import SE3
from .extractor import SuperPointExtractor
from utils.geo_utils import *
from .layer import *
from utils.DBA_utils import *

class VO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.extractor = SuperPointExtractor()
        self.DT = compute_delaunay_edges  # Delaunay Triangulation

        # 1. 시각 및 기하 매칭 모듈
        self.stereo_matcher = EpipolarCrossAttention(feature_dim=256)
        self.temporal_matcher = TemporalCrossAttention(feature_dim=256)
        
        # 2. 깊이 초기화 및 특징 융합 (우리가 설계한 핵심!)
        self.stereo_depth_mod = StereoDepthModule(cfg.focal, cfg.baseline, feature_dim=256)
        self.geo_bottleneck = GeometricBottleneck(visual_dim=768, geo_dim=64, out_dim=256)

        # 3. 그래프 최적화 및 업데이트 블록
        # Bottleneck을 거쳐 256차원으로 고정된 입력을 받습니다.
        self.update_block = GraphUpdateBlock(hidden_dim=256)
        
        # 4. 수치적 솔버
        self.DBA = DBASolver()
        self.DBA_Updater = PoseDepthUpdater(min_depth=0.1, max_depth=100.0)
        self.log_lmbda = nn.Parameter(torch.tensor(-4.6)) # 초기 댐핑 lmbda = 0.01

    def forward(self, batch, iters=8, gt_guide=None):
        """
        [Inputs]
        - batch: 데이터셋에서 제공하는 dict
            - 'images': [B, 4, 1, H, W] (Lt, Rt, Lt1, Rt1 순서)
            - 'calib': [B, 4] (fx, fy, cx, cy)
            - 'node_features': (Train 전용) 미리 추출된 특징 [B, 4, N, 256]
            - 'kpts': (Train 전용) 미리 추출된 좌표 [B, 4, N, 2]
        - iters: 반복 최적화 횟수 (기본 8회)
        - gt_guide: (Optional) 초기 포즈 가이드 [B, 7]
        """
        device = self.log_lmbda.device
        calib = batch['calib'].to(device)
        B = calib.shape[0]
        
        # --- [Step 1] Feature & Graph Extraction ---
        if self.training:
            # 학습 모드: 연산 속도를 위해 전처리된 특징 사용
            node_feats_all = batch['node_features'].to(device) # [B, 4, N, 256]
            kpts_all = batch['kpts'].to(device)               # [B, 4, N, 2]
            edges_list = batch['edges']                        # Delaunay 에지 리스트
            edge_attr_list = batch['edge_attr']                # 에지 속성 리스트
        # else:
        #     # 추론 모드: 원본 이미지로부터 실시간 추출
        #     images = batch['images'].to(device) # [B, 4, 1, H, W]
        #     B, V, _, H, W = images.shape
        #     # SuperPoint 등을 통한 특징점 및 디스크립터 추출 로직 (생략/유지)
        #     kpts_all, node_feats_all, edges_list, edge_attr_list = self.extractor(images)

        # 4개 프레임 분리 (Lt: 현재, Rt: 현재 우측, Lt1: 다음 프레임)
        f_Lt, kpts_Lt = node_feats_all[:, 0], kpts_all[:, 0]
        f_Rt, kpts_Rt = node_feats_all[:, 1], kpts_all[:, 1]
        f_Lt1, kpts_Lt1 = node_feats_all[:, 2], kpts_all[:, 2]

        # --- [Step 2] Stereo Initialization (Initial Depth) ---
        # 1. Epipolar Attention을 통한 매칭 및 시차 계산
        f_stereo, init_disp, conf_s = self.stereo_matcher(f_Lt, f_Rt, kpts_Lt, kpts_Rt)
        
        # 2. 스테레오 모듈을 통한 초기 깊이 추정 및 기하 특징(s_geo) 생성
        # nodes_L_up: [B, N, 544], s_geo: [B, N, 32]
        nodes_L_up, curr_depth = self.stereo_depth_mod(f_Lt, f_stereo, init_disp, conf_s)

        # --- [Step 3] Iterative Refinement Loop ---
        curr_pose = SE3.Identity(B, device=device)
        h = torch.zeros((B, kpts_Lt.shape[1], 256), device=device) # GRU Hidden State
        
        poses_history = []
        depths_history = []
        weights_history = []  

        for i in range(iters):
            if i == 0 and gt_guide is not None:
                curr_pose = SE3.InitFromVec(gt_guide)

            # 1. Adaptive Temporal Matching
            # 현재 예측된 포즈(curr_pose)로 Lt의 점들을 Lt1으로 재투영
            kpts_t1_pred = project_kpts(kpts_Lt, curr_depth, curr_pose, calib)
            
            # 윈도우 서치를 통해 시간적 매칭 특징 및 기하 특징(t_geo) 생성
            f_temp, t_geo, flow_res, conf_t = self.temporal_matcher(
                f_Lt, f_Lt1, kpts_t1_pred, kpts_Lt1, iter_idx=i
            )

            # 2. Geometric Bottleneck (차원 융합: 768+64 -> 256)
            # v_feat: [Lt, Stereo, Temporal] 결합
            v_feat = torch.cat([f_Lt, f_stereo, f_temp], dim=-1) # [B, N, 768]
            # g_feat: [Stereo_Geo, Temporal_Geo] 결합 (각 32차원 인코딩됨)
            g_feat = torch.cat([nodes_L_up[..., -32:], t_geo], dim=-1) # [B, N, 64]
            
            x_fused = self.geo_bottleneck(v_feat, g_feat) # [B, N, 256]

            # 3. Graph Update Block (GAT + GRU)
            # x_fused는 이제 수치적으로 매우 안정적인 256차원 공간 인지 벡터입니다.
            h, r, w, a_p, a_d = self.update_block(h, flow_res, x_fused, edges_list, edge_attr_list) 
            
            # 4. DBA Solver & Update
            J_p, J_d = compute_projection_jacobian(kpts_Lt, curr_depth, calib)
            lmbda = torch.exp(self.log_lmbda)
            
            delta_pose, delta_depth = self.DBA(r, w, J_p, J_d, lmbda)
            curr_pose, curr_depth = self.DBA_Updater(curr_pose, curr_depth, delta_pose, delta_depth, a_p, a_d)

            poses_history.append(curr_pose)
            depths_history.append(curr_depth)

        return poses_history, depths_history, weights_history[-1] # 최종 결과와 시각화용 가중치 반환
    
def project_kpts(kpts_t0, depth_t0, pose_t0t1, calib):
    """
    kpts_t0: [B, N, 2] - 현재 프레임의 2D 특징점 (u, v)
    depth_t0: [B, N, 1] - 현재 프레임의 깊이 (Z)
    pose_t0t1: lietorch.SE3 객체 - t0에서 t1으로의 상대 포즈 변화
    calib: [B, 4] - 카메라 내상수 [fx, fy, cx, cy]
    """
    B, N, _ = kpts_t0.shape
    device = kpts_t0.device
    
    fx, fy, cx, cy = calib[:, 0:1], calib[:, 1:2], calib[:, 2:3], calib[:, 3:4]

    # 1. 2D 픽셀 좌표 -> 3D 카메라 좌표계 (Back-projection)
    # x = (u - cx) * Z / fx
    # y = (v - cy) * Z / fy
    u, v = kpts_t0[..., 0:1], kpts_t0[..., 1:2]
    x = (u - cx.unsqueeze(1)) * depth_t0 / fx.unsqueeze(1)
    y = (v - cy.unsqueeze(1)) * depth_t0 / fy.unsqueeze(1)
    z = depth_t0
    
    pts_3d_t0 = torch.cat([x, y, z], dim=-1) # [B, N, 3]

    # 2. SE3 포즈 변환 (t0 좌표계 -> t1 좌표계)
    # lietorch의 SE3 객체는 [B, N, 3] 형태의 포인트 클라우드 변환을 지원합니다.
    # pose_t0t1가 [B] 크기라면 노드 개수(N)만큼 확장해서 적용해야 할 수도 있습니다.
    pts_3d_t1 = pose_t0t1[:, None] * pts_3d_t0 # [B, N, 3]
    # 3. 3D 카메라 좌표계 -> 2D 픽셀 좌표계 (Projection)
    # u' = x' * fx / z' + cx
    # v' = y' * fy / z' + cy
    x1, y1, z1 = pts_3d_t1[..., 0:1], pts_3d_t1[..., 1:2], pts_3d_t1[..., 2:3]
    
    # zero-division 방지
    z1 = torch.clamp(z1, min=0.1)
    
    u1 = (x1 * fx.unsqueeze(1) / z1) + cx.unsqueeze(1)
    v1 = (y1 * fy.unsqueeze(1) / z1) + cy.unsqueeze(1)
    
    kpts_t1_pred = torch.cat([u1, v1], dim=-1) # [B, N, 2]
    
    return kpts_t1_pred