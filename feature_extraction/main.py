import scipy.io
from pathlib import Path
import numpy as np
from datetime import datetime
import os
import functools
from joblib import Parallel, delayed
import pandas as pd
import json

import sys

from sklearn.preprocessing import StandardScaler

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from feature_extraction.SequenceTupleException import SequenceTupleException
from feature_extraction.create_feature_names import create_feature_names_as_list
from feature_extraction.feature_extraction_functions import signal_tga, signal_max, signal_range, signal_iqr, \
    signal_std, signal_idr, signal_mean, signal_mad, signal_tmax, signal_median, \
    signal_tgm, signal_min, signal_sgm, signal_sga, signal_area, signal_area_min_max, \
    signal_first_derivative, signal_second_derivative, mean_hr, mean_rr, \
    rms_diffs, signal_zero_crossing, signal_var, \
    mean_value_first_diff, max_to_min_peak_value_ratio, signal_rms, signal_mean_local_max, \
    signal_mean_local_min, \
    mean_absolute_values, std_absolute_values, \
    rms_of_successive_diffs, std_of_successive_diffs, signal_split_equal_part_mean, signal_split_equal_part_std, \
    signal_split_equal_part_var, signal_area_min_max_ratio, signal_p2pmv, mean_absolute_value_first_diff, \
    mean_absolute_value_second_diff
from feature_extraction.helpers import process_dataset_preprocessing, compute_start_indices, \
    process_dataset_extract_features, replace_or_remove_occurring_nans_in_data_set
from feature_extraction.preprocessing import butter_bandpass_partial, butter_bandpass_partial_ecg, \
    reset_bad_labels_for_editing, cut_start, editing_subject_data, repair_faulty_labels


channel_preproc_config_dict = {
    'corrugator': [butter_bandpass_partial],
    'zygomaticus': [butter_bandpass_partial],
    'trapezius': [butter_bandpass_partial],
    'scl': [],
    'ecg': [butter_bandpass_partial_ecg]
}



#######################################
#### ANNPR Features incl. features of Werner/Walter et al.
#######################################

# functions to apply on the derivatives of a signal
functions_to_apply_on_derivatives = [signal_mean, signal_median, signal_min, signal_max, signal_range, signal_std,
                                     signal_iqr, signal_idr, signal_mad, signal_tmax, signal_tgm, signal_tga,
                                     signal_sgm, signal_sga, signal_area, signal_area_min_max,
                                     signal_zero_crossing, signal_split_equal_part_mean, signal_split_equal_part_std, signal_var,
                                     max_to_min_peak_value_ratio, signal_rms, mean_absolute_values, std_absolute_values,
                                     # signal_split_equal_part_mean, signal_split_equal_part_std,
                                     signal_split_equal_part_var]

# first derivative
compute_st_order_derivative_features = functools.partial(signal_first_derivative,
                                                         func_list=functions_to_apply_on_derivatives)

compute_st_order_derivative_features.__name__ = "first_derivative_features"

# second derivative
compute_nd_order_derivative_features = functools.partial(signal_second_derivative,
                                                         func_list=functions_to_apply_on_derivatives)

compute_nd_order_derivative_features.__name__ = "second_derivative_features"

# feature extraction functions
channel_feature_extr_config_dict_time_domain = {
    'corrugator': [signal_mean, signal_median, signal_min, signal_max, signal_range, signal_std,
                   signal_iqr, signal_idr, signal_mad, signal_tmax, signal_tgm, signal_tga,
                   signal_sgm, signal_sga, signal_area, signal_area_min_max,
                   signal_zero_crossing, signal_split_equal_part_mean, signal_split_equal_part_std,
                   signal_var, mean_value_first_diff, max_to_min_peak_value_ratio, signal_rms, signal_p2pmv,
                   signal_mean_local_max, signal_mean_local_min, mean_absolute_values, std_absolute_values,
                   mean_absolute_value_first_diff, mean_absolute_value_second_diff,
                   signal_split_equal_part_var,
                   compute_st_order_derivative_features,
                   compute_nd_order_derivative_features,
                   # newly added features after ANNPR evalutation
                   signal_area_min_max_ratio],
    'zygomaticus': [signal_mean, signal_median, signal_min, signal_max, signal_range, signal_std,
                    signal_iqr, signal_idr, signal_mad, signal_tmax, signal_tgm, signal_tga,
                    signal_sgm, signal_sga, signal_area, signal_area_min_max,
                    signal_zero_crossing, signal_split_equal_part_mean, signal_split_equal_part_std,
                    signal_var, mean_value_first_diff, max_to_min_peak_value_ratio, signal_rms, signal_p2pmv,
                    signal_mean_local_max, signal_mean_local_min, mean_absolute_values, std_absolute_values,
                    mean_absolute_value_first_diff, mean_absolute_value_second_diff,
                    signal_split_equal_part_var,
                    compute_st_order_derivative_features,
                    compute_nd_order_derivative_features,
                    # newly added features after ANNPR evalutation
                    signal_area_min_max_ratio],
    'trapezius': [signal_mean, signal_median, signal_min, signal_max, signal_range, signal_std,
                  signal_iqr, signal_idr, signal_mad, signal_tmax, signal_tgm, signal_tga,
                  signal_sgm, signal_sga, signal_area, signal_area_min_max,
                  signal_zero_crossing, signal_split_equal_part_mean, signal_split_equal_part_std,
                  signal_var, mean_value_first_diff, max_to_min_peak_value_ratio, signal_rms, signal_p2pmv,
                  signal_mean_local_max, signal_mean_local_min, mean_absolute_values, std_absolute_values,
                  mean_absolute_value_first_diff, mean_absolute_value_second_diff,
                  signal_split_equal_part_var,
                  compute_st_order_derivative_features,
                  compute_nd_order_derivative_features,
                  # newly added features after ANNPR evalutation
                  signal_area_min_max_ratio],
    'scl': [signal_mean, signal_median, signal_min, signal_max, signal_range, signal_std,
            signal_iqr, signal_idr, signal_mad, signal_tmax, signal_tgm, signal_tga,
            signal_sgm, signal_sga, signal_area, signal_area_min_max,
            signal_zero_crossing, signal_split_equal_part_mean, signal_split_equal_part_std, signal_var,
            mean_value_first_diff, max_to_min_peak_value_ratio, signal_rms, mean_absolute_values, std_absolute_values,
            mean_absolute_value_first_diff, mean_absolute_value_second_diff,
            signal_split_equal_part_var,
            compute_st_order_derivative_features,
            compute_nd_order_derivative_features,
            # newly added features after ANNPR evalutation
            signal_area_min_max_ratio],
    'ecg': [signal_mean, signal_median, signal_min, signal_max, signal_range, signal_std,
            signal_iqr, signal_idr, signal_mad, signal_tmax, signal_tgm, signal_tga,
            signal_sgm, signal_sga, signal_area, signal_area_min_max,
            rms_diffs, signal_zero_crossing, signal_split_equal_part_mean, signal_split_equal_part_std,
            signal_var, mean_value_first_diff, max_to_min_peak_value_ratio, signal_rms, signal_p2pmv, signal_mean_local_max,
            signal_mean_local_min, mean_absolute_values, std_absolute_values, mean_absolute_value_first_diff,
            mean_absolute_value_second_diff, rms_of_successive_diffs, std_of_successive_diffs,
            signal_split_equal_part_var,
            compute_st_order_derivative_features,
            compute_nd_order_derivative_features,
            # newly added features after ANNPR evalutation
            mean_rr, mean_hr, signal_area_min_max_ratio]
}


def compute_end_idx_stimuli(current_i, start_indices, minimal_length, sh, extension=0, shift=0):
    current_idx = start_indices[current_i]
    # check for end
    if current_i >= len(start_indices):
        # this case should not happen at all due to the criterions which are set for valid stimuli sequences
        # but just in case (back up etc. you know) - this case is handle here
        if current_idx + minimal_length + extension + shift >= sh:
            start_next_stimuli = sh
        # enough sample points at the end
        else:
            # add + 1 since last idx is exclusive
            start_next_stimuli = current_idx + minimal_length + extension + shift + 1
    # not the end of the sequence
    else:
        # if the current stimuli is not the last one, just walk until the index of the next stimuli
        start_next_stimuli = start_indices[current_i + 1] + extension + shift
    return start_next_stimuli


def extract_pain_stimuli(subj_id, current_i, start_indices, data, current_label, extension=0, tonic_length=60000,
                         phasic_length=3900, phasic=True, heat_phasic=True, correction=1000, shift=0):

    # label of the current data slice
    slice_label = current_label
    # start of the current data slice (which has a shift included if start indices was computed with a shift set
    start_current_stimuli = start_indices[current_i] + shift
    # differentiation between phasic and heat stimuli
    if phasic:
        # compute end index of current phasic stimuli
        start_next_stimuli = compute_end_idx_stimuli(current_i, start_indices,
                                                     phasic_length, data.shape[0], extension=extension, shift=shift)
        if heat_phasic:
            # heat phasic
            data_slice = data[start_current_stimuli:start_next_stimuli, :]

        else:  # electro phasic -> apply correction
            data_slice = data[start_current_stimuli + correction:start_next_stimuli, :]

    # tonic stimuli
    else:
        # compute end index of current tonic stimuli
        start_next_stimuli = compute_end_idx_stimuli(current_i, start_indices, tonic_length, data.shape[0],
                                                     extension=extension, shift=shift)
        # no correction here since all tonic stimuli are in the same length present
        data_slice = data[start_current_stimuli:start_next_stimuli, :]

    print('Data slice has shape: {} with label {}'.format(data_slice.shape, slice_label))

    return subj_id, start_indices[current_i], np.copy(data_slice), slice_label


def extract_baseline_stimuli(subj_id, current_i, start_indices, data, this_baseline_label,
                             extension=0, correction=1000,  shift=0):
    start_current_stimuli = start_indices[current_i] + shift + correction
    current_baseline_length = start_indices[current_i] - start_indices[current_i - 1] - correction
    # this case should not happen at all due to the criterions which are set for valid stimuli sequences
    # but just in case (back up etc. you know) - this case is handle here

    # shift is already applied for the start of the current stimuli
    if start_current_stimuli + current_baseline_length + extension >= data.shape[0]:
        start_next_stimuli = data.shape[0]
    # enough sample points at the end
    else:
        # add + 1 since last idx is exclusive
        start_next_stimuli = start_current_stimuli + current_baseline_length + extension

    data_slice = data[start_current_stimuli:start_next_stimuli, :]
    print('Data slice has shape: {} with label {}'.format(data_slice.shape, this_baseline_label))
    return subj_id, start_indices[current_i], np.copy(data_slice), this_baseline_label


# wrapper function for multiprocessing
def process_slice(seq_tuple,
                  channel_features_dict=None,
                  default_o=None,
                  ch_names=None):
    # seq_tuple -> (subject_id, current_data, current_label)
    subj_id, glob_idx, c_data, c_label = seq_tuple
    print("Current data slice start idx: {}, size: {}, label: {}".format(glob_idx, c_data.shape, c_label))
    sys.stdout.flush()
    # print("Current data slice size: {}".format(c_data.shape))
    if ch_names is None:
        ch_names = {'corrugator': 0, 'zygomaticus': 1, 'trapezius': 2, 'scl': 3, 'ecg': 4}
    if channel_features_dict is None:
        channel_features_dict = []
    if default_o is None:
        default_o = ['corrugator', 'zygomaticus', 'trapezius', 'scl', 'ecg']
    current_features = process_dataset_extract_features(c_data,
                                                        channel_features_dict=channel_features_dict,
                                                        default_order=default_o,
                                                        default_channel_names=ch_names)

    current_features = np.concatenate((current_features, [c_label]))
    current_features = np.concatenate(([subj_id], current_features))
    return current_features


def split_tonic_sequence(slice_tup, split_length):
    subj_id, start_idx, data_slice, slice_label = slice_tup
    slice_tup_list = []

    # number of splits
    s = int(data_slice.shape[0] / split_length)
    for split_idx in range(0, s):
        phasic_slice = np.copy(data_slice[split_idx * split_length: split_idx * split_length + split_length, :])
        # collect splits
        slice_tup_list.append((subj_id, start_idx + (split_idx * split_length), phasic_slice, slice_label))
    return slice_tup_list


# do not use subjects according to Werner et al. "Twofold-Multimodal Pain Recognition with the X-ITE Pain Database"
do_not_use = ["S001.mat", "S014.mat", "S023.mat", "S024.mat", "S025.mat", "S030.mat", "S059.mat", "S121.mat"]
# no electro stimuli present
do_not_use = ["S028.mat"] + do_not_use


if __name__ == '__main__':

    FE_CONFIG_FILE_PATH = Path('../fe_config.json')
    fe_config = {}
    with open(FE_CONFIG_FILE_PATH, 'r') as fd:
        fe_config = json.load(fd)

    x_ite_ed_bio_path = Path(fe_config['x_ite_ed_bio_path'])
    path_to_store_dataset = fe_config['path_to_store_dataset']
    dataset_file_name = fe_config['dataset_file_name']

    n_jobs = fe_config['n_jobs']
    parallel_backend = fe_config['parallel_backend']

    # used for offset parameter in compute_start_indices
    shift_phasic_heat = 2000
    shift_phasic_electro = 0

    tonic_split_sequence = True
    tonic_split_heat_length = 4000  # resulting in 15 phasic heat sequences of length 4 s
    tonic_split_electro_length = 5000  # resulting in 12 phasic electric sequences of length 5 s

    # used for offset window after applied stimulus
    ext_heat = 500
    ext_electro = 0

    correction_heat_electro = 1000

    # tolerances for subject correction
    accepted_labels = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    phasic_baseline_length_tolerance = 8000
    phasic_length_tolerance = 3900
    tonic_baseline_length_tolerance = 62000
    tonic_length_tolerance = 60000
    shift_tolerance = 5000

    # labels for the different baselines
    base_line_label_tonic_heat = 200  # as an indicator for tonic baseline feature vectors
    base_line_label_tonic_electro = -200
    base_line_label_phasic_heat = 100
    base_line_label_phasic_electro = -100

    phasic_stimuli_labels = [-3, -2, -1, 1, 2, 3]
    phasic_heat_stimuli_labels = [1, 2, 3]
    tonic_stimuli_labels = [-6, -5, -4, 4, 5, 6]
    tonic_heat_stimuli_labels = [4, 5, 6]
    tonic_electro_stimuli_labels = [-4, -5, -6]
    baseline_label = 0

    extract_b_after_tonic_heat = 4
    extract_b_after_tonic_electro = -4
    extract_b_after_phasic_heat = 1
    extract_b_after_phasic_electro = -1


    if not os.path.exists(path_to_store_dataset):
        os.makedirs(path_to_store_dataset)
    file_list = list(x_ite_ed_bio_path.glob('**/*.mat'))
    # filter subjects out which will not used in the later tasks
    usable_files = [x for x in file_list if x.name not in do_not_use]
    channel_names = {'corrugator': 0, 'zygomaticus': 1, 'trapezius': 2, 'scl': 3, 'ecg': 4}

    # for testing purpose only
    # subjects_file_list = list(filter(lambda x: '042' in x.name or '043' in x.name , usable_files))

    subjects_file_list = usable_files
    subjects_file_list.sort(key=lambda x: int(x.name.split('.')[0].split('S')[1]))
    default_order = ['corrugator', 'zygomaticus', 'trapezius', 'scl', 'ecg']

    # iterate over all files / subjects and accumulate the results in a list since the
    # number of samples is not known before (only number of subjects)
    accumulator_samples = []
    print("Start processing subjects: ")
    start = datetime.now()
    for k, subject_path in enumerate(subjects_file_list):
        # print("\rSubject: {}".format(Path(subject_path).stem), end='')
        print("Subject: {} ({} of {})".format(Path(subject_path).stem, k+1, len(subjects_file_list)))
        # load data from current path / subject
        subject_data = scipy.io.loadmat(subject_path)
        subject_labels = subject_data['stimuli']
        subject_data = subject_data['data']
        subject_id = int(Path(subject_path).stem.split('S')[-1])

        # edit data by removing sequences which contain bad labels
        subject_labels_reset = reset_bad_labels_for_editing(subject_labels, set_to_label=-11, filter_for_label=[-10])

        subject_data_repaired, subject_labels_repaired = repair_faulty_labels(subject_data, subject_labels_reset,
                                                                              accepted_labels,
                                                                              phasic_baseline_length_tolerance,
                                                                              phasic_length_tolerance,
                                                                              tonic_baseline_length_tolerance,
                                                                              tonic_length_tolerance,
                                                                              shift_tolerance)

        subject_data_shaved, subject_labels_shaved = cut_start(subject_data_repaired, subject_labels_repaired)
        subject_data_edited, subject_labels_edited = editing_subject_data(subject_data_shaved,
                                                                          subject_labels_shaved,
                                                                          label_to_cut_off=-11)

        processed_data_subject = process_dataset_preprocessing(subject_data_edited, channel_preproc_config_dict)

        # do slicing by labels and feature extraction here - which is a shift
        stimuli_start_indices = compute_start_indices(subject_labels_edited, shift=0)
        print("Size of start_indices: {}".format(len(stimuli_start_indices)))

        collection_idx_slices = []
        for i, idx in enumerate(stimuli_start_indices):
            # current label might be changed if a phasic baseline occurs
            current_label = subject_labels_edited[idx]
            current_data = None

            # phasic stimuli
            if current_label in phasic_stimuli_labels:
                if current_label in phasic_heat_stimuli_labels:
                    slice_tup = extract_pain_stimuli(subject_id, i, stimuli_start_indices, processed_data_subject,
                                                     current_label, extension=ext_heat,
                                                     tonic_length=tonic_length_tolerance,
                                                     phasic_length=phasic_length_tolerance, phasic=True,
                                                     heat_phasic=True, correction=0,
                                                     shift=shift_phasic_heat)
                else:
                    slice_tup = extract_pain_stimuli(subject_id, i, stimuli_start_indices, processed_data_subject,
                                                     current_label, extension=ext_electro,
                                                     tonic_length=tonic_length_tolerance,
                                                     phasic_length=phasic_length_tolerance, phasic=True,
                                                     heat_phasic=False, correction=correction_heat_electro,
                                                     shift=shift_phasic_electro)
            # tonic stimuli
            elif current_label in tonic_stimuli_labels:
                slice_tup = extract_pain_stimuli(subject_id, i, stimuli_start_indices, processed_data_subject,
                                                 current_label, extension=0, tonic_length=tonic_length_tolerance,
                                                 phasic_length=phasic_length_tolerance, phasic=False, heat_phasic=False,
                                                 correction=0, shift=0)
                if tonic_split_sequence:
                    # call function to split data
                    if slice_tup[3] in tonic_electro_stimuli_labels:
                        slice_tup = split_tonic_sequence(slice_tup, tonic_split_electro_length)
                    elif slice_tup[3] in tonic_heat_stimuli_labels:
                        slice_tup = split_tonic_sequence(slice_tup, tonic_split_heat_length)
                    else:
                        raise SequenceTupleException('Label of sequence unknown, split not possible. '
                                                     'Label {} is not part of tonic electro (labels: {}) nor '
                                                     'tonic heat (labels: {}) '
                                                     'stimuli.'.format(slice_tup[3], tonic_electro_stimuli_labels,
                                                                       tonic_heat_stimuli_labels))

            # only baseline is left
            else:
                # this case should not happen due to modification of each sequence start
                if i == 0:
                    continue
                else:
                    current_shift = 0
                    current_ext = 0
                    current_correction = 0
                    tonic_split = False
                    current_baseline_split_length = 0
                    prev_label = subject_labels_edited[stimuli_start_indices[i - 1]]
                    print('Prev label baseline: {}'.format(prev_label))
                    if prev_label == extract_b_after_phasic_electro:
                        current_label = base_line_label_phasic_electro  # e. g. -100
                        # set electro specific shift which is used for pain stimuli as well
                        current_shift = shift_phasic_electro
                        current_ext = ext_electro
                        current_correction = correction_heat_electro
                    elif prev_label == extract_b_after_phasic_heat:
                        current_label = base_line_label_phasic_heat  # e. g. 100
                        # set heat specific shift which is used for pain stimuli as well
                        current_shift = shift_phasic_heat
                        current_ext = ext_heat
                    elif prev_label == extract_b_after_tonic_electro:
                        current_label = base_line_label_tonic_electro  # e. g. -200
                        tonic_split = True
                        current_baseline_split_length = tonic_split_electro_length
                    elif prev_label == extract_b_after_tonic_heat:
                        current_label = base_line_label_tonic_heat  # e. g. 200
                        tonic_split = True
                        current_baseline_split_length = tonic_split_heat_length
                    else:
                        continue

                    # call function to extract baseline
                    slice_tup = extract_baseline_stimuli(subject_id, i, stimuli_start_indices,
                                                         processed_data_subject, current_label, extension=current_ext,
                                                         correction=current_correction, shift=current_shift)
                    if tonic_split_sequence and tonic_split:
                        slice_tup = split_tonic_sequence(slice_tup, current_baseline_split_length)
            # add the splits if present
            if type(slice_tup) == list:
                collection_idx_slices = collection_idx_slices + slice_tup
            # add the tuple itself if no split of the sequence occurred
            else:
                collection_idx_slices.append(slice_tup)

        results = Parallel(n_jobs=n_jobs,
                           backend=parallel_backend)(delayed(process_slice)(tup,
                                                                            channel_feature_extr_config_dict_time_domain,
                                                                            default_order, channel_names)
                                                     for tup in
                                                     collection_idx_slices)

        accumulator_samples = accumulator_samples + results

    # store features in matrix
    # path_repaired_nans_json = os.path.join(path_to_store_dataset, dataset_file_name + '_repaired_rows_cols_nan.json')

    data_set = np.vstack(accumulator_samples)
    data_set, repaired_dict_nans = replace_or_remove_occurring_nans_in_data_set(data_set, strategy='mean',
                                                                                filter_function=np.isnan)
    data_set, repaired_dict_infs = replace_or_remove_occurring_nans_in_data_set(data_set, strategy='mean',
                                                                                filter_function=np.isinf)

    config_dict = {
        'n_jobs': n_jobs,
        'parallel_backend': parallel_backend,

        'do_not_use_subjects': do_not_use,

        # used for offset parameter in compute_start_indices
        'shift_phasic_heat': shift_phasic_heat,
        'shift_phasic_electro': shift_phasic_electro,

        'tonic_split_sequence': tonic_split_sequence,  # flag to indicate that tonic sequences were split
        'tonic_split_heat_length': tonic_split_heat_length,
        'tonic_split_electro_length': tonic_split_electro_length,
        # used for offset window after applied stimulus
        'ext_heat': ext_heat,
        'ext_electro': ext_electro,

        'correction_heat_electro': correction_heat_electro,

        # tolerances for subject correction
        'accepted_labels': accepted_labels,
        'phasic_baseline_length_tolerance': phasic_baseline_length_tolerance,
        'phasic_length_tolerance': phasic_length_tolerance,
        'tonic_baseline_length_tolerance': tonic_baseline_length_tolerance,
        'tonic_length_tolerance': tonic_length_tolerance,
        'shift_tolerance': shift_tolerance,

        'tonic_heat_stimuli_labels': tonic_heat_stimuli_labels,
        'tonic_electro_stimuli_labels': tonic_electro_stimuli_labels,

        # labels for the different baselines
        'base_line_label_tonic_heat': base_line_label_tonic_heat,
        'base_line_label_tonic_electro': base_line_label_tonic_electro,
        'base_line_label_phasic_heat': base_line_label_phasic_heat,
        'base_line_label_phasic_electro': base_line_label_phasic_electro,

        'phasic_stimuli_labels': phasic_stimuli_labels,
        'phasic_heat_stimuli_labels': phasic_heat_stimuli_labels,
        'tonic_stimuli_labels': tonic_stimuli_labels,
        'baseline_label': baseline_label,

        'extract_b_after_tonic_heat': extract_b_after_tonic_heat,
        'extract_b_after_tonic_electro': extract_b_after_tonic_electro,
        'extract_b_after_phasic_heat': extract_b_after_phasic_heat,
        'extract_b_after_phasic_electro': extract_b_after_phasic_electro
    }

    # build proprocessing function name dict since partial function etc. are not serializable
    preproc_info_dict = {}
    for curr_ch_preproc in default_order:
        preproc_info_dict[curr_ch_preproc] = [func.__name__ for func in channel_preproc_config_dict[curr_ch_preproc]]

    info_dict = {
        'config': config_dict,
        'preprocessing_functions': preproc_info_dict,
        'nans_repaired': repaired_dict_nans,
        'infs_repaired': repaired_dict_infs
    }

    data_set_real = np.zeros(data_set.shape)
    # convert complex numbers to real numbers
    for col in range(data_set.shape[1]):
        data_set_real[:, col] = np.abs(data_set[:, col]) if np.any(data_set[:, col].imag != 0) else np.real(data_set[:, col])

    stop = datetime.now()
    duration = stop - start
    info_dict['config']['timestamp_start'] = start.strftime('%Y-%m-%d_%H-%M')
    info_dict['config']['timestamp_stop'] = stop.strftime('%Y-%m-%d_%H-%M')
    info_dict['config']['timestamp_duration'] = '{}'.format(duration)

    # file name of the new created data set
    dataset_file_name = dataset_file_name + '_{}'.format(stop.strftime('%Y-%m-%d_%H-%M'))
    json_file_name = os.path.join(path_to_store_dataset, dataset_file_name + '_config.json')

    dataset_file_name_csv = dataset_file_name + '.csv'
    info_dict['config']['dataset_file_name'] = dataset_file_name_csv
    path_data_set = os.path.join(path_to_store_dataset, dataset_file_name_csv)

    print("\nDuration time: {}".format(duration))
    print("Dataset shape: {}".format(data_set.shape))
    print("Dataset type: {}".format(type(data_set)))

    feature_names = create_feature_names_as_list(channel_feature_extr_config_dict_time_domain, default_order)
    info_dict['features'] = feature_names

    with open(json_file_name, 'w') as fd:
        json.dump(info_dict, fd, indent=4)

    # np.save(path_data_set + '.npy', data_set_real, allow_pickle=False, fix_imports=False)

    cols = ["subj_id"] + feature_names + ["label"]
    df = pd.DataFrame(data_set_real, columns=cols)

    # subject specific standardization (z-score: zero mean with unit variance)
    unique_ids = np.unique(df.iloc[:, 0].values)
    mod_feature_vec = []
    for s_id in unique_ids:
        scaler = StandardScaler()
        idx = np.where(df['subj_id'] == s_id)[0]
        vals = df.iloc[idx, 1:-1].to_numpy()
        tmp_res = scaler.fit_transform(vals)
        df.iloc[idx, 1:-1] = tmp_res

    df.to_csv(path_data_set)
