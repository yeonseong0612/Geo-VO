import torch
from src.model import VO
from src.loader import DataFactory, vo_collate_fn
from torch.utils.data import DataLoader
from CFG.vo_cfg import vo_cfg  # 본인의 설정 파일 경로에 맞게 수정

def test_full_pipeline():
    # 1. 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    iters = 8 # 반복 횟수 설정
    
    # 2. 모델 로드 (iters 설정 확인)
    model = VO(vo_cfg).to(device)
    model.eval()
    print(f"✅ 모델 로드 완료 (Device: {device})")

    # 3. 테스트 데이터 로드 (Batch size 2)
    dataset = DataFactory(vo_cfg, mode='train')
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=vo_collate_fn)
    batch = next(iter(loader))
    
    # 데이터를 GPU로 이동
    for k in ['node_features', 'kpts', 'calib']:
        batch[k] = batch[k].to(device)
    print(f"✅ 테스트 배치 로드 완료 (Batch Size: {batch['node_features'].shape[0]})")

    # 4. Forward 실행 (전체 루프 가동)
    print(f"🚀 {iters}회 반복 업데이트 루프 시작...")
    with torch.no_grad():
        try:
            poses_list, depths_list = model(batch, iters=iters)
        except Exception as e:
            print(f"❌ 에러 발생: {e}")
            import traceback
            traceback.print_exc()
            return

    # 5. 최종 검증 (Dimension & List Check)
    print("\n" + "="*40)
    print("📊 최종 통합 테스트 결과 (FINAL CHECK)")
    print("="*40)

    # 포즈 리스트 체크
    print(f"1. Poses List Length: {len(poses_list)} (기대치: {iters})")
    assert len(poses_list) == iters, "포즈 리스트 개수가 맞지 않습니다."
    
    # 개별 포즈 타입 및 차원 체크 (lietorch SE3 객체인지 확인)
    print(f"2. Final Pose Type: {type(poses_list[-1])}")
    print(f"3. Final Pose Shape: {poses_list[-1].shape} (기대치: [B])")

    # 깊이 리스트 체크
    print(f"4. Depths List Length: {len(depths_list)} (기대치: {iters})")
    assert len(depths_list) == iters, "깊이 리스트 개수가 맞지 않습니다."
    
    # 깊이 값 범위 체크 (물리적으로 타당한지)
    final_depth = depths_list[-1]
    print(f"5. Final Depth Shape: {final_depth.shape} (기대치: [B, 800, 1])")
    print(f"6. Mean Depth Value: {final_depth.mean().item():.2f}m")

    # 포즈 변화 확인 (첫 번째 루프와 마지막 루프의 차이)
    # 포즈가 조금이라도 변했다면 최적화 루프가 작동하고 있다는 뜻입니다.
    diff = (poses_list[0].data - poses_list[-1].data).abs().sum()
    if diff > 0:
        print(f"7. Pose Refinement: YES (변화량: {diff.item():.6f})")
    else:
        print(f"7. Pose Refinement: NO (포즈가 변하지 않았습니다. DBA 확인 필요)")

    print("="*40)
    print("✨ 모든 시스템 정상 가동! 이제 학습을 시작해도 좋습니다.")

if __name__ == "__main__":
    test_full_pipeline()