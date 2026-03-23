import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'lasot'

# trackers.extend(trackerlist(name='ostrack', parameter_name='ce_256_r1', dataset_name=dataset_name, run_ids=None, display_name='ce_256_r1'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ce_256_r2', dataset_name=dataset_name, run_ids=None, display_name='ce_256_r2'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='ce_256_r3', dataset_name=dataset_name, run_ids=None, display_name='ce_256_r3'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='evit_256_r1', dataset_name=dataset_name, run_ids=None, display_name='evit_256_r1'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='evit_256_r2', dataset_name=dataset_name, run_ids=None, display_name='evit_256_r2'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='evit_256_r3', dataset_name=dataset_name, run_ids=None, display_name='evit_256_r3'))

# trackers.extend(trackerlist(name='ostrack', parameter_name='tome_256_r8', dataset_name=dataset_name, run_ids=None, display_name='tome_256_r8'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='tome_256_r14', dataset_name=dataset_name, run_ids=None, display_name='tome_256_r14'))
# trackers.extend(trackerlist(name='ostrack', parameter_name='tome_256_r20', dataset_name=dataset_name, run_ids=None, display_name='tome_256_r20'))

trackers.extend(trackerlist(name='ostrackcmp', parameter_name='utptrack_384_r7_soft', dataset_name=dataset_name, run_ids=None, display_name='utptrack_384_r7_soft'))
# trackers.extend(trackerlist(name='ostrackcmp', parameter_name='utptrack_256_r8', dataset_name=dataset_name, run_ids=None, display_name='utptrack_256_r8'))
# trackers.extend(trackerlist(name='ostrackcmp', parameter_name='utptrack_256_r9', dataset_name=dataset_name, run_ids=None, display_name='utptrack_256_r9'))

dataset = get_dataset(dataset_name)
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'), skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
