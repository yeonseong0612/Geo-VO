import torch
from lietorch import SE3
import numpy as np
from scipy.spatial.transform import Rotation as R

def verify_conversion():
    print("--- 🔍 lietorch 정밀 구조 분석 (v2) ---")
    
    # 1. 입력 생성: [x, y, z, qx, qy, qz, qw]
    # Z축 90도 회전 쿼터니언: [0, 0, 0.7071, 0.7071]
    t_in = torch.tensor([10.0, 20.0, 30.0])
    q_in = torch.tensor([0.0, 0.0, 0.7071, 0.7071])
    vec7_input = torch.cat([t_in, q_in], dim=0).float()
    
    # 2. SE3 객체 생성
    pose_obj = SE3.InitFromVec(vec7_input.unsqueeze(0))
    
    # 3. 내부 데이터(.data) 확인
    # 여기서 shape이 [1, 7]인지 [1, 8]인지가 핵심입니다.
    raw_data = pose_obj.data.squeeze(0)
    print(f"가져온 SE3.data Shape: {raw_data.shape}")
    print(f"가져온 SE3.data 값: {raw_data}")

    # 4. 분해 테스트 (AttributeError 방지용 안전한 접근)
    # lietorch SE3는 보통 .translation()과 .data[:, 3:7]로 나뉩니다.
    t_out = pose_obj.translation().squeeze(0)
    
    # 쿼터니언 속성명 확인 (버전마다 다름: .quat, .quaternion, .data[:, 3:7])
    try:
        q_out = pose_obj.quat().squeeze(0)
        method_name = "quat()"
    except:
        try:
            q_out = pose_obj.unit_quaternion().squeeze(0)
            method_name = "unit_quaternion()"
        except:
            # 메서드가 없으면 내부 데이터에서 직접 슬라이싱 (가장 확실)
            q_out = raw_data[3:7] 
            method_name = "data[3:7] slicing"

    print("-" * 50)
    print(f"추출 방법: {method_name}")
    print(f"추출된 Translation: {t_out}")
    print(f"추출된 Quaternion : {q_out}")

    # 5. [중요] 다시 합치기 테스트
    recombined = torch.cat([t_out, q_out], dim=0)
    
    # 6. 최종 정합성 체크
    is_same = torch.allclose(vec7_input, recombined, atol=1e-4)
    
    print("-" * 50)
    if is_same:
        print("✅ 결론: 분해 후 합치기(cat)가 안전합니다!")
        print("   순서가 [x, y, z, qx, qy, qz, qw]로 완벽히 유지됩니다.")
    else:
        print("❌ 결론: 순서가 뒤섞였습니다! 값을 비교해 보세요.")
        print(f"원래 입력: {vec7_input}")
        print(f"다시 합침: {recombined}")

if __name__ == "__main__":
    verify_conversion()