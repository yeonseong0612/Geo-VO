import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.loader import DataFactory, vo_collate_fn
# 실제 모델 클래스와 설정 파일을 임포트하세요 (파일 이름에 맞춰 수정)
from src.model import VO 

def test_model_unit():
    # 1. 테스트용 설정 (로더 테스트와 동일)
    class Config:
        proj_home = './'
        odometry_home = '/home/yskim/projects/vo-labs/data/kitti_odometry/'
        precomputed_dir = './data/precomputed'
        color_subdir = 'datasets/sequences/'
        poses_subdir = 'poses/'
        calib_subdir = 'datasets/sequences/'
        traintxt = 'train.txt'
        trainsequencelist = ['00']
        
        # 모델 관련 하이퍼파라미터 예시
        hidden_dim = 128
        iters = 8 # DBA 반복 횟수

    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📡 Testing on device: {device}")

    try:
        # 2. 로더 초기화 (Train 모드 - NPZ 로드)
        dataset = DataFactory(cfg, mode='train')
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=vo_collate_fn)
        batch = next(iter(loader))
        print("배치 데이터 준비 완료")

        # 3. 모델 초기화
        model = VO(cfg).to(device)
        model.train() # 학습 모드
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        # 배치 데이터를 GPU로 이동
        # (딕셔너리 내부의 텐서들을 이동시키는 유틸리티 함수가 있으면 좋습니다)
        input_data = {
            'node_features': batch['node_features'].to(device),
            'edges': batch['edges'].to(device),
            'edge_attr': batch['edge_attr'].to(device),
            'masks': batch['masks'].to(device),
            'intrinsics': batch['clib'].to(device)
        }
        gt_pose = batch['rel_pose'].to(device) # [B, 7]

        # 4. Forward Pass
        print("Forward pass 시작...")
        pred_pose = model(input_data) # 모델 아웃풋 형태에 따라 수정 필요
        
        print(f"Forward 성공! 출력 차원: {pred_pose.shape}")

        # 5. Loss & Backward Pass
        # 단순 MSE로 먼저 테스트 (나중에 Geodesic Loss 등으로 교체)
        loss = torch.nn.functional.mse_loss(pred_pose, gt_pose)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Backward 성공! Loss: {loss.item():.6f}")
        print("\n모델 검사 최종 합격: 데이터 로더부터 역전파까지 정상 작동합니다.")

    except Exception as e:
        print(f"모델 검사 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_unit()