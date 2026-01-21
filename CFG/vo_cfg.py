from easydict import EasyDict
import os

vo_cfg = EasyDict()

vo_cfg.odometry_home = '/home/jnu-ie/Dataset/kitti_odometry/'
vo_cfg.proj_home = '/home/jnu-ie/kys/Geo-VO/'
vo_cfg.model = 'GEO-VO'

vo_cfg.logdir = os.path.join(vo_cfg.proj_home, 'checkpoint', vo_cfg.model)

vo_cfg.color_subdir = 'data_odometry_color/dataset/sequences/'
vo_cfg.calib_subdir = 'data_odometry_calib/dataset/sequences/'
vo_cfg.poses_subdir = 'data_odometry_poses/dataset/poses/'

vo_cfg.precomputed_dir = os.path.join(vo_cfg.proj_home, 'gendata/precomputed')

vo_cfg.traintxt = 'train.txt'
vo_cfg.valtxt = 'val.txt'

vo_cfg.trainsequencelist = ['00', '01', '02', '03', '04', '05', '06', '07', '08']
vo_cfg.valsequencelist = ['09', '10']

vo_cfg.baseline = 0.54
vo_cfg.num_cpu = 12            
vo_cfg.batchsize = 8           
vo_cfg.learning_rate = 1e-4
vo_cfg.maxepoch = 60

vo_cfg.iters = 8                
vo_cfg.weight_decay = 1e-4     
vo_cfg.MultiStepLR_milstone = [30, 50]
vo_cfg.MultiStepLR_gamma = 0.1

vo_cfg.log_interval = 10        