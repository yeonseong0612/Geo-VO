from easydict import EasyDict
import os

vo_cfg = EasyDict()

# 1. Directory - 실제 사용하는 경로만 남김
vo_cfg.odometry_home = '/home/jnu-ie/Dataset/kitti_odometry/'
vo_cfg.proj_home = '/home/jnu-ie/kys/GEO-VO/'
vo_cfg.model = 'Geo-vo'

vo_cfg.logdir = os.path.join(vo_cfg.proj_home, 'checkpoint', vo_cfg.model)
vo_cfg.color_subdir = 'data_odometry_color/dataset/sequences/'
vo_cfg.calib_subdir = 'data_odometry_calib/dataset/sequences/'
vo_cfg.poses_subdir = 'data_odometry_poses/dataset/poses/'

# 2. Train 리스트
vo_cfg.trainsequencelist = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
vo_cfg.valsequencelist = ['09', '10']

# 3. Hyperparameters - 성능 최적화
vo_cfg.num_cpu = 4             # 데이터 로딩 속도 향상
vo_cfg.batchsize = 4           # GPU 메모리 상황에 맞게 조절
vo_cfg.learning_rate = 1e-4    # 2e-3은 VO 학습에 너무 클 수 있어 1e-4 권장 (오타 수정)
vo_cfg.maxepoch = 60
vo_cfg.MultiStepLR_milstone = [30, 50]
vo_cfg.MultiStepLR_gamma = 0.1