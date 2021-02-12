"""
Adapted from Allen from:
https://github.com/AllenInstitute/ecephys_spike_sorting.git
"""
import click
import os

from .. import io
from .. import ksort_postprocessing
from ..wrappers import calculate_metrics


@click.command()
@click.option('--kilosort_folder', default=None, help='kilosort_folder to read from and write to')
@click.option('--do_parallel', default=True, type=bool, help='Parallel or not, 0 or 1')
@click.option('--do_silhouette', default=True, type=bool, help='do_silhouette or not, 0 or 1')
@click.option('--do_drift', default=True, type=bool, help='do_drift or not, 0 or 1')
@click.option('--do_pc_features', default=True, type=bool, help='do_pc_features or not, 0 or 1')
def cli(kilosort_folder=None, do_parallel=True, do_pc_features=True, do_silhouette=True, do_drift=True, fs=3e4):
    """ Calculate metrics for all units on one probe"""
    # kilosort_folder = '~/res_ss_full/res_ss/tcloop_train_m022_1553627381_'
    if kilosort_folder is None:
        kilosort_folder = os.getcwd()
    print(f'Running cluster_quality in folder {kilosort_folder}')
    if do_pc_features:
        do_include_pcs = True
    else:
        do_include_pcs = False
    print(do_pc_features, do_silhouette, do_drift)
    (the_spike_times, the_spike_clusters, the_spike_templates, the_templates, the_amplitudes, the_unwhitened_temps,
     the_channel_map, the_cluster_ids, the_cluster_quality,
     the_pc_features, the_pc_feature_ind) = io.load_kilosort_data(kilosort_folder,
                                                                  fs,
                                                                  False,
                                                                  include_pcs=do_include_pcs)

    try:
        (the_spike_times, the_spike_clusters, the_spike_templates,
         the_amplitudes, the_pc_features,
         the_overlap_matrix) = ksort_postprocessing.remove_double_counted_spikes(the_spike_times,
                                                                                 the_spike_clusters,
                                                                                 the_spike_templates,
                                                                                 the_amplitudes,
                                                                                 the_channel_map,
                                                                                 the_templates,
                                                                                 the_pc_features,
                                                                                 sample_rate=fs)
    except IndexError as e:  # IndexError
        print(e)
        print('Cannot remove overlapping spikes due to error above ')

    calculate_metrics(the_spike_times, the_spike_clusters, the_spike_templates,
                      the_amplitudes, the_pc_features, the_pc_feature_ind,
                      output_folder=kilosort_folder,
                      do_pc_features=do_pc_features,
                      do_silhouette=do_silhouette,
                      do_drift=do_drift,
                      do_parallel=do_parallel)
    return 0


# Launch this file and drop into debug if needed
if __name__ == '__main__':
    cli()
