from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.prj_dir = '/home/hk-project-p0022189/tum_yvc3016/haowu/UTPTrack/UTPTrack-S-0812'
    settings.save_dir = settings.prj_dir
    # settings.network_path = settings.prj_dir + '/test/networks'    # Where tracking networks are stored.
    settings.network_path = settings.prj_dir + '/checkpoints'    # Where tracking networks are stored.
    settings.result_plot_path = settings.prj_dir + '/test/result_plots'
    settings.results_path = settings.prj_dir + '/test/tracking_results'    # Where to store tracking results
    settings.segmentation_path = settings.prj_dir + '/test/segmentation_results'


    settings.data_dir = '/home/hk-project-p0022189/tum_yvc3016/haowu/datasets/tracking-datasets'
    settings.got10k_path = settings.data_dir + '/got10k'
    settings.got10k_lmdb_path = '' # '/got10k_lmdb'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_path = settings.data_dir + '/lasot'
    settings.lasotlang_path = settings.data_dir + '/lasot'
    settings.lasot_lmdb_path = ''   # '/lasot_lmdb'
    settings.lasot_extension_subset_path = settings.data_dir + '/lasot_extension_subset'
    settings.trackingnet_path = settings.data_dir + '/trackingnet'

    settings.otblang_path = settings.data_dir + '/otb_lang'
    settings.tnl2k_path = settings.data_dir + '/tnl2k/test'

    settings.nfs_path = '' # '/nfs'
    settings.otb_path = '' # '/OTB2015'
    settings.tc128_path = ''        # '/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.uav_path = ''          # '/UAV123'
    settings.vot_path = ''          # '/VOT2019'
    settings.youtubevos_dir = ''
    settings.davis_dir = ''

    return settings

