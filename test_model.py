import torch

def test_calib_slicing():
    # 1. 실제 상황 설정 (DDP 배치가 1일 때, 전체 프레임은 4개)
    B_actual = 1
    num_views = 4
    total_calib_rows = B_actual * num_views # 결과: 4
    
    # 더미 calib 생성 [4, 4] -> 각 행이 [fx, fy, cx, cy]
    # 각 프레임마다 구분하기 쉽게 fx 값을 다르게 설정
    # 0번: Lt, 1번: Rt, 2번: Lt1, 3번: Rt1
    calib = torch.tensor([
        [450.0, 450.0, 320.0, 240.0], # Lt0 (우리가 필요한 것)
        [451.0, 451.0, 320.0, 240.0], # Rt0
        [452.0, 452.0, 320.0, 240.0], # Lt1
        [453.0, 453.0, 320.0, 240.0]  # Rt1
    ])

    print(f"==> Raw Calib Shape: {calib.shape}") # [4, 4]

    # 2. 기존 방식 (에러 발생 원인)
    focal_wrong = calib[:, 0:1] 
    print(f"\n[!] 기존 방식 focal shape: {focal_wrong.shape}") 
    # 결과: [4, 1] -> 노드 배치 1과 맞지 않음 (Expected 1 but got 4)

    # 3. 수정 방식 (Lt 프레임만 정확히 추출)
    # 4개씩 묶인 데이터에서 첫 번째(Lt)만 가져오기
    focal_correct = calib[::4, 0:1] 
    print(f"==> 수정 방식 focal shape: {focal_correct.shape}")
    print(f"    추출된 fx 값: {focal_correct.squeeze().item()} (Lt0의 fx와 일치해야 함)")

    # 4. 브로드캐스팅 시뮬레이션 (StereoDepthModule 내부)
    N = 100 # 노드 개수
    init_disp = torch.randn(B_actual, N, 1) # [1, 100, 1]
    
    print(f"\n==> Broadcasting Test:")
    try:
        # fB_expanded = [1, 1, 1]
        fB_expanded = focal_correct.unsqueeze(1) 
        inv_depth = init_disp / (fB_expanded + 1e-6)
        print(f"    [Success] inv_depth shape: {inv_depth.shape}")
    except Exception as e:
        print(f"    [Failure] 연산 에러: {e}")

    # 5. 여러 배치일 때 테스트 (B=2)
    print("\n==> Multi-Batch Test (B=2):")
    calib_B2 = torch.cat([calib, calib + 10], dim=0) # [8, 4]
    focal_B2 = calib_B2[::4, 0:1] # [2, 1]
    print(f"    calib_B2 shape: {calib_B2.shape}")
    print(f"    focal_B2 shape: {focal_B2.shape} (기대값: [2, 1])")

if __name__ == "__main__":
    test_calib_slicing()