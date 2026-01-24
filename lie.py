import torch
from lietorch import SE3

def test_update_direction():
    # 1. 초기 포즈: 배치 크기 1인 Identity 생성
    # lietorch는 batch_shape를 명시적으로 (1,) 처럼 넣어줘야 합니다.
    curr_pose = SE3.Identity(1, device='cpu') 
    
    # 2. 업데이트량: x축으로 1m 이동 (vx=1.0)
    # lietorch의 exp 인자는 [tx, ty, tz, qx, qy, qz, qw] 순서가 아니라 
    # tangent vector인 [tx, ty, tz, wx, wy, wz] (6차원) 형태여야 합니다.
    delta_pose = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device='cpu')
    a_p = 1.0
    
    # 3. 업데이트 수행 (왼쪽 곱)
    delta_SE3 = SE3.exp(a_p * delta_pose)
    new_pose = delta_SE3 * curr_pose
    
    # 4. 결과 해석
    # lietorch SE3의 데이터는 [tx, ty, tz, qx, qy, qz, qw] 순서로 저장됩니다.
    data = new_pose.data
    tx, ty, tz = data[0, 0], data[0, 1], data[0, 2]
    
    print(f"Update Result (Translation) -> x: {tx:.4f}, y: {ty:.4f}, z: {tz:.4f}")
    
    if tx > 0:
        print("✅ 결과: '왼쪽 곱'이 로컬 x축 방향으로 전진을 의미합니다.")
    else:
        print("❌ 결과: 왼쪽 곱이 의도와 다르게 작동합니다.")

if __name__ == "__main__":
    test_update_direction()