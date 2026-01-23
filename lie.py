import torch
from lietorch import SE3

# 1. 확연히 구분되는 값 설정
# Translation 의도: 10, 20, 30
# Quaternion 의도: 0, 0, 0, 1 (Identity)
t_val = [10.0, 20.0, 30.0]
q_val = [0.0, 0.0, 0.0, 1.0]

# 케이스 A: [Translation, Quaternion] 순서로 생성
vec_a = torch.tensor(t_val + q_val).float().unsqueeze(0)
pose_a = SE3.InitFromVec(vec_a)

# 케이스 B: [Quaternion, Translation] 순서로 생성
vec_b = torch.tensor(q_val + t_val).float().unsqueeze(0)
pose_b = SE3.InitFromVec(vec_b)

print("="*50)
print(f"입력 데이터 (A) [T, Q]: {vec_a.tolist()}")
print(f"결과 (A) translation(): {pose_a.translation().tolist()}")

print("-"*50)

print(f"입력 데이터 (B) [Q, T]: {vec_b.tolist()}")
print(f"결과 (B) translation(): {pose_b.translation().tolist()}")
print("="*50)

# 회전값 확인 (객체의 rotation 속성이나 matrix 활용)
print(f"회전 행렬 (B):\n{pose_b.matrix().tolist()[0]}")