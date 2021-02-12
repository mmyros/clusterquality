import numpy as np

from cluster_quality import wrappers
from tests.test_wrappers import download_and_load

np.random.seed(1000)


def test_calculate_pc_features():
    (base_path, path_expected, spike_times, spike_clusters, spike_templates, templates, amplitudes,
     unwhitened_temps, channel_map, cluster_ids, cluster_quality, pc_features, pc_feature_ind
     ) = download_and_load(subsample=500)

    df = wrappers.calculate_metrics(spike_times, spike_clusters, spike_templates, amplitudes, pc_features,
                                    pc_feature_ind,
                                    output_folder=None, do_parallel=True,
                                    do_pc_features=True, do_silhouette=False, do_drift=False)
    # df.to_csv(path_expected / 'pc_features.csv', index=False)  # Uncomment this if results must change
    # pd.testing.assert_frame_equal(df.round(1), pd.read_csv(path_expected / 'pc_features.csv').round(1),
    #                               check_dtype=False)

    for col in df.columns:
        assert not df[col].isna().all(), f' Column {col} is all nan'
