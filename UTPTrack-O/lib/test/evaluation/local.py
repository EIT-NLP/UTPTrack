from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.prj_dir = '/hkfs/home/project/hk-project-p0022189/tum_yvc3016/haowu/code/UTPTrack/UTPTrack-O-0807-rebuttal-got10k'
    settings.save_dir = settings.prj_dir
    settings.network_path = settings.prj_dir + '/checkpoints'    # Where tracking networks are stored.
    settings.result_plot_path = settings.save_dir + '/test/result_plots'
    settings.results_path = settings.save_dir + '/test/tracking_results'    # Where to store tracking results
    settings.segmentation_path = settings.save_dir + '/test/segmentation_results'

    settings.data_dir = '/hkfs/home/project/hk-project-p0022189/tum_yvc3016/haowu/datasets/tracking-datasets'

    settings.lasot_path = settings.data_dir + '/lasot'
    settings.lasot_lmdb_path = ''
    
    settings.lasot_extension_subset_path = settings.data_dir + '/lasot_extension_subset'
        
    settings.trackingnet_path = settings.data_dir + '/trackingnet'

    settings.got10k_path = settings.data_dir + '/got10k'
    settings.got10k_lmdb_path = ''


    settings.lasotlang_path = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = ''
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = ''
    settings.tpl_path = ''
    settings.uav_path = ''
    settings.vot18_path = ''
    settings.vot22_path = ''
    settings.vot_path = ''
    settings.youtubevos_dir = ''
    settings.davis_dir = ''    
    
    return settings

