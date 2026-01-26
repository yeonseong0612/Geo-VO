import torch
from lietorch import SE3
import numpy as np

def verify_update_order():
    # 1. 현재 포즈 (차 위치): x축으로 10m 가 있는 상태
    # [tx, ty, tz, qx, qy, qz, qw]
    curr_pose = SE3.InitFromVec(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    
    # 2. 업데이트량 (delta): 제자리에서 왼쪽으로 90도 회전 (y축 기준)
    # 쿼터니언 [0, 0.707, 0, 0.707] 은 y축 90도 회전
    delta_vec = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.7071, 0.0, 0.7071])
    delta_SE3 = SE3.InitFromVec(delta_vec)

    # CASE A: Left Multiplication (Delta * T) -> World Frame 기준
    # "세상을 원점 기준으로 돌려버림"
    left_update = delta_SE3 * curr_pose
    
    # CASE B: Right Multiplication (T * Delta) -> Local Frame 기준
    # "차를 현재 위치에서 제자리 회전시킴"
    right_update = curr_pose * delta_SE3

    print("=== [SE3 업데이트 순서 검증] ===")
    print(f"초기 위치: {curr_pose.translation()[0].numpy()}")
    print("-" * 30)
    print(f"Left Multi (World) 후 위치:  {left_update.translation()[0].numpy()}")
    print(" -> 해석: 원점(0,0,0)을 기준으로 10m 떨어진 차를 돌려버림 (x=10 이 z=10 으로 이동)")
    print("-" * 30)
    print(f"Right Multi (Local) 후 위치: {right_update.translation()[0].numpy()}")
    print(" -> 해석: 현재 위치(10,0,0)는 유지하면서 차의 방향만 바뀜 (자전)")

if __name__ == "__main__":
    verify_update_order()