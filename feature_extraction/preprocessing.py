import numpy as np
import functools
from scipy.signal import butter, lfilter
from .helpers import compute_start_indices


# copy paste from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_params(cut, fs, order=5, f_type='band'):
    nyq = 0.5 * fs

    if len(cut) == 1:
        normal_cutoff = cut[0] / nyq
        b, a = butter(order, normal_cutoff, btype=f_type, analog=False)
        return b, a
    elif len(cut) == 2:
        low = cut[0] / nyq
        high = cut[1] / nyq
        b, a = butter(order, [low, high], btype=f_type, analog=False)
        return b, a
    else:
        print(
            'Wrong input for cutoff frequency! Input was {} but needs to be [low or high] or [low, high].'.format(cut))
        return None, None


def butter_filter(data, cut, fs, order=5, f_type='band'):
    b, a = butter_params(cut, fs, order=order, f_type=f_type)
    y = lfilter(b, a, data)
    return y


# applied filters (imported by main.py)
# butter_bandpass_partial = functools.partial(butter_filter, cut=(0.5, 25), fs=1000, order=3, f_type='band')
butter_bandpass_partial = functools.partial(butter_filter, cut=(20, 250), fs=1000, order=3, f_type='band')
# butter_lowpass_partial = functools.partial(butter_filter, cut=[50], fs=1000, order=3, f_type='low')
butter_bandpass_partial_ecg = functools.partial(butter_filter, cut=(0.1, 25), fs=1000, order=3, f_type='band')

# provide names for partial functions
butter_bandpass_partial.__name__ = 'butter_bandpass_3rd_order_cut_20-250'
butter_bandpass_partial_ecg.__name__ = 'butter_bandpass_3rd_order_cut_0.1-25'


def reset_bad_labels_for_editing(stimuli, set_to_label=-11, filter_for_label=None):
    if filter_for_label is None:
        filter_for_label = [-10]
    # check if there are label == -11 and label == -10 present
    # if so, set all with label == -10 to label = -11
    # cut off slices which do have label == -11 as stimuli.
    # This is totally okay since both are not usable for us
    # therefore, it does not matter how to reset these labels
    # as long as we cut off those.
    for label in filter_for_label:
        idx_current = np.where(stimuli == label)[0]
        stimuli[idx_current] = set_to_label
    return stimuli


def editing_subject_data(data, stimuli, label_to_cut_off=-11):
    start_indices = compute_start_indices(stimuli_labels=stimuli, shift=0)
    # start_indices_incl_labels = zip(start_indices, stimuli[start_indices])
    # get all start indices of label == -11
    # length of start_indices_pos_11 should be equal to length of start_indices, right?
    start_indices_pos_11 = np.where(stimuli[start_indices] == label_to_cut_off)[0]

    indices_to_delete = []
    for i in start_indices_pos_11:
        if stimuli[start_indices[i - 1]] == 0:
            pre_edit_offset = -2
        else:
            pre_edit_offset = -1

        if i + 1 >= len(start_indices):
            # to indicate, that a selection till the end of the sequence is necessary
            post_edit_offset = np.inf
        elif stimuli[start_indices[i + 1]] == 0:
            post_edit_offset = +2
        else:
            post_edit_offset = +3

        if post_edit_offset == np.inf:
            indices_to_delete += list(range(start_indices[i + pre_edit_offset], stimuli.shape[0]))
        # if the current start index equals the last element in stimuli, add only this index to delete list
        elif i == stimuli.shape[0] - 1:
            indices_to_delete += list(start_indices[i])
        # if the current start index with added post_edit_offset is larger than the size of stimuli, then
        # add only indices from pre_edit to the end of stimuli
        elif i + post_edit_offset >= stimuli.shape[0] - 1:
            indices_to_delete += list(range(start_indices[i + pre_edit_offset], stimuli.shape[0]))
        # apply offset without fearing out of bounds
        else:
            indices_to_delete += list(range(start_indices[i + pre_edit_offset], start_indices[i + post_edit_offset]))

    modified_data = np.delete(data, indices_to_delete, axis=0)
    modified_stimuli = np.delete(stimuli, indices_to_delete).reshape(-1,)

    return modified_data, modified_stimuli


def cut_start(data, stimuli, valid_labels=tuple(list(range(-6, 0)) + list(range(1, 7)))):
    start_indices = compute_start_indices(stimuli_labels=stimuli, shift=0)

    label_sequence = [0, 0]
    for i in start_indices:
        # does not need to be 0 and if it starts with a stimuli: ignore it
        # since we do not know if it is a complete stimuli sequence
        if i == 0:
            label_sequence[0] = stimuli[i]
        else:
            label_sequence[0] = label_sequence[1]
            label_sequence[1] = stimuli[i]

        if label_sequence[0] == 0 and label_sequence[1] in valid_labels:
            # delete dat shit
            indices_to_delete = list(range(0, i))

            modified_data = np.delete(data, indices_to_delete, axis=0)
            modified_stimuli = np.delete(stimuli, indices_to_delete)

            return modified_data, modified_stimuli

        else:
            continue

    return None


def tolerance_check(current_label, prev_label, label_length,
                    phasic_baseline_length_tolerance=8000,
                    phasic_stimuli_length_tolerance=4000,
                    tonic_baseline_length_tolerance=62000,
                    tonic_stimuli_length_tolerance=60000):

    if current_label in [-10, -11]:
        return False
    elif prev_label is None:
        if current_label == 0:
            return False
        elif np.abs(current_label) in [1, 2, 3]:
            return label_length < phasic_stimuli_length_tolerance
        else:
            return label_length < tonic_stimuli_length_tolerance
    else: # previous label exists
        if current_label == 0:
            if np.abs(prev_label) in [1, 2, 3]:
                return label_length < phasic_baseline_length_tolerance
            else:
                return label_length < tonic_baseline_length_tolerance

        elif np.abs(current_label) in [1, 2, 3]:
            return label_length < phasic_stimuli_length_tolerance
        else:
            return label_length < tonic_stimuli_length_tolerance


def repair_faulty_labels(data, stimuli,
                         valid_labels=(-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6),
                         phasic_baseline_length_tolerance=8000,
                         phasic_stimuli_length_tolerance=4000,
                         tonic_baseline_length_tolerance=62000,
                         tonic_stimuli_length_tolerance=60000,
                         shift_tolerance=5000):

    start_indices = compute_start_indices(stimuli, shift=0)

    indices_marked_for_repair = []  # collect indices of all faulty sequences

    for i, idx in enumerate(start_indices):

        current_label = stimuli[idx]

        if i >= len(start_indices) - 1:
            stimuli_length = len(stimuli) - idx
        else:
            stimuli_length = start_indices[i + 1] - idx

        if i == 0:
            if tolerance_check(current_label, None, stimuli_length,
                               phasic_baseline_length_tolerance,
                               phasic_stimuli_length_tolerance,
                               tonic_baseline_length_tolerance,
                               tonic_stimuli_length_tolerance):
                indices_marked_for_repair += list(range(idx, start_indices[i + 1]))
            
        elif i + 1 >= len(start_indices):
            prev_label = stimuli[start_indices[i - 1]]
            if current_label == 0:
                if prev_label in [1, 4]:
                    if prev_label == 1:
                        # since it is a relevant baseline -> check if it has the necessary length
                        # if not, remove baseline and the previous stimuli
                        if stimuli_length < phasic_baseline_length_tolerance:
                            indices_marked_for_repair += list(range(start_indices[i - 1], len(stimuli)))
                    else:  # if it is not a baseline which follows a phasic stimuli then check for tonic tolerance
                        if stimuli_length < tonic_baseline_length_tolerance:
                            indices_marked_for_repair += list(range(start_indices[i - 1], len(stimuli)))
                else:
                    # not able to perform shift at the end
                    if stimuli_length < shift_tolerance:
                        indices_marked_for_repair += list(range(start_indices[i - 1], len(stimuli)))
            else:  # not a label == 0 -> cut-off label since sequence has to end with a baseline
                indices_marked_for_repair += list(range(start_indices[i], len(stimuli)))
            
            continue

        elif i > 0:  # ensure previous label exists
            prev_label = stimuli[start_indices[i - 1]]

            # no baseline between two stimuli
            if current_label != 0 and prev_label != 0 and not (current_label == -11 or prev_label == -11):
                if i + 2 >= len(start_indices):
                    indices_marked_for_repair += list(range(start_indices[i - 1], len(stimuli)))
                else:
                    indices_marked_for_repair += list(range(start_indices[i - 1], start_indices[i + 2]))

            elif current_label in valid_labels and prev_label == 0:
                if tolerance_check(current_label, prev_label, stimuli_length,
                                   phasic_baseline_length_tolerance,
                                   phasic_stimuli_length_tolerance,
                                   tonic_baseline_length_tolerance,
                                   tonic_stimuli_length_tolerance):
                    if i + 2 >= len(start_indices):
                        indices_marked_for_repair += list(range(idx, len(stimuli)))
                    else:
                        indices_marked_for_repair += list(range(idx, start_indices[i + 2]))

            elif current_label == 0 and prev_label in valid_labels:
                if tolerance_check(current_label, prev_label, stimuli_length,
                                   phasic_baseline_length_tolerance,
                                   phasic_stimuli_length_tolerance,
                                   tonic_baseline_length_tolerance,
                                   tonic_stimuli_length_tolerance):
                    if i + 1 >= len(start_indices):
                        indices_marked_for_repair += list(range(start_indices[i - 1], len(stimuli)))
                    else:
                        indices_marked_for_repair += list(range(start_indices[i - 1], start_indices[i + 1]))
    
    indices_marked_for_repair = np.asarray(np.unique(indices_marked_for_repair), dtype=np.int64)

    modified_data = np.delete(data, indices_marked_for_repair, axis=0)
    modified_stimuli = np.delete(stimuli, indices_marked_for_repair)

    return modified_data, modified_stimuli

def identity_helper(data):
    return data
