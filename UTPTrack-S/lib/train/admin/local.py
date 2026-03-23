class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/hkfs/home/project/hk-project-p0022189/tum_yvc3016/haowu/UTPTrack-S-0812'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'

        self.data_dir = '/hkfs/home/project/hk-project-p0022189/tum_yvc3016/haowu/datasets/tracking-datasets'

        self.lasot_dir = self.data_dir + '/lasot'
        self.lasot_lmdb_dir = self.data_dir + '/lasot_lmdb'
        
        self.got10k_dir = self.data_dir + '/got10k/train'
        self.got10k_lmdb_dir = self.data_dir + '/got10k_lmdb'
        
        self.trackingnet_dir = self.data_dir + '/trackingnet'
        self.trackingnet_lmdb_dir = self.data_dir + '/trackingnet_lmdb'
        
        self.coco_dir = self.data_dir + '/coco'
        self.coco_lmdb_dir = self.data_dir + '/coco_lmdb'

        self.vasttrack_dir = self.data_dir + '/vasttrack'
        self.depthtrack_dir = self.data_dir + '/depthtrack/train'
        self.lasher_dir = self.data_dir + '/lasher/trainingset'
        self.visevent_dir = self.data_dir + '/visevent/train'
        self.tnl2k_dir = self.data_dir + '/tnl2k/train'
        self.otb99_dir = self.data_dir + '/otb_lang'


        self.refcoco_dir = ''       # '/refcoco'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet1k_dir = ''    # '/imagenet1k'
        self.imagenet22k_dir = ''   # '/imagenet22k'
        self.imagenet_dir = ''      # '/vid'
        self.imagenet_lmdb_dir = '' # '/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''