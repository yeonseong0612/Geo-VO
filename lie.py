import torch
from lietorch import SE3

# 1. 초기 포즈: y축으로 90도 회전된 상태 (카메라가 오른쪽을 보고 있음)
# [tx, ty, tz, qx, qy, qz, qw]
initial_vec = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.707, 0.0, 0.707]]).cuda()
curr_pose = SE3.InitFromVec(initial_vec)

# 2. 증분: x축으로 +1m 이동
delta_pose = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).cuda()
delta_SE3 = SE3.exp(delta_pose)

# 3. 비교
left_update = delta_SE3 * curr_pose  # World 기준 업데이트
right_update = curr_pose * delta_SE3 # Local 기준 업데이트

print(f"Left Update (World x-axis):\n{left_update.data}")
print(f"Right Update (Local x-axis):\n{right_update.data}")