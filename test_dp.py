import torch
import torch.nn as nn
from lietorch import SE3
import lietorch

class MockVO(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.randn(1))
    
    def forward(self, x):
        # 실제 모델처럼 SE3 객체와 일반 텐서를 리턴
        # [Iteration=8, Batch=각 GPU별 배치, 7]
        batch_size = x.shape[0]
        poses = SE3(torch.randn(8, batch_size, 7).cuda())
        weights = torch.randn(8, batch_size, 1).cuda()
        return poses, weights

def test():
    device = torch.device("cuda")
    model = MockVO().to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    # 배치 사이즈 4 (GPU 2개면 각각 2개씩 처리)
    dummy_input = torch.randn(4, 3, 224, 224).cuda()
    
    print("🚀 모델 실행...")
    outputs = model(dummy_input)
    
    poses_h = outputs[0]
    print(f"1. poses_h 전체 타입: {type(poses_h)}")
    
    if isinstance(poses_h, (list, tuple, map)):
        poses_list = list(poses_h)
        print(f"2. 리스트 변환 후 개수: {len(poses_list)}")
        print(f"3. 리스트 내부 첫 요소 타입: {type(poses_list[0])}")
        
        # 병합 테스트
        try:
            combined = lietorch.cat(poses_list, dim=1)
            print(f"4. 병합 성공! 최종 Shape: {combined.shape}")
        except Exception as e:
            print(f"4. 병합 실패: {e}")

if __name__ == "__main__":
    test()