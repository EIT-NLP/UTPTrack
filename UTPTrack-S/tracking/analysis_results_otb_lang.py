import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'otb99_lang'

trackers.extend(trackerlist(name='sutrack', parameter_name='sutrack_b224', dataset_name=dataset_name, run_ids=None, display_name='sutrack_b224'))
trackers.extend(trackerlist(name='sutrackcmp', parameter_name='ceatetta_b224_soft', dataset_name=dataset_name, run_ids=None, display_name='ceatetta_b224_soft'))
trackers.extend(trackerlist(name='sutrackcmp', parameter_name='ceatettamma_b224_d', dataset_name=dataset_name, run_ids=None, display_name='ceatettamma_b224_d'))

dataset = get_dataset(dataset_name)

# print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'), force_evaluation=True)

print_per_sequence_results(trackers, dataset, dataset_name, merge_results=True, force_evaluation=True)
