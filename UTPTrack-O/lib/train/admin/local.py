class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/hkfs/home/project/hk-project-p0022189/tum_yvc3016/haowu/code/UTPTrack/UTPTrack-O-0807-rebuttal-got10k'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'

        self.data_dir = '/hkfs/home/project/hk-project-p0022189/tum_yvc3016/haowu/datasets/tracking-datasets'
        
        self.lasot_dir = self.data_dir + '/lasot'
        self.lasot_lmdb_dir = self.data_dir + '/lasot_lmdb'
        
        self.got10k_dir = self.data_dir + '/got10k/train'
        self.got10k_lmdb_dir = self.data_dir + '/got10k_lmdb'
        self.got10k_val_dir = self.data_dir + '/got10k/val'
        
        self.trackingnet_dir = self.data_dir + '/trackingnet'
        self.trackingnet_lmdb_dir = self.data_dir + '/trackingnet_lmdb'
        
        self.coco_dir = self.data_dir + '/coco'
        self.coco_lmdb_dir = self.data_dir + '/coco_lmdb'
                
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''

