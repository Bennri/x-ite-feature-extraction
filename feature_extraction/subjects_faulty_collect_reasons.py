import numpy as np
import scipy.io
from datetime import datetime
from pathlib import Path
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from feature_extraction.helpers import compute_start_indices


# TODO set path to dataset
x_ite_ed_bio_path = Path("")
result_file_path = Path("reasons_faulty_subjects_raw.txt")

file_list = list(x_ite_ed_bio_path.glob('**/*.mat'))

if 'clean' in x_ite_ed_bio_path.name:
    file_list.sort(key=lambda x: int(x.name.split('.')[0].split('S')[1].split('_')[0]))
else:
    file_list.sort(key=lambda x: int(x.stem.split('S')[1]))  # side-effect


def calculate_label_length(current_i, current_idx, calculated_start_indices, labels):
    if current_i >= len(calculated_start_indices) - 1:
        current_label_length = len(labels) - current_idx
    else:
        current_label_length = calculated_start_indices[current_i + 1] - current_idx
    return current_label_length


def append_calculated_lengths(current_label_length, current_label_func, dict_to_append):
    if current_label_func in dict_to_append.keys():
        dict_to_append[current_label_func].append(current_label_length)


accepted_labels = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
phasic_baseline_length_tolerance = 8000
phasic_length_tolerance = 3900
tonic_baseline_length_tolerance = 62000
tonic_length_tolerance = 60000
shift_tolerance = 5000

rejected_subjects = []

start = datetime.now()
for k, subject_path in enumerate(file_list):
    # load data from current path / subject
    subject_data = scipy.io.loadmat(subject_path)
    subject_labels = subject_data['stimuli'].reshape(-1, )
    if 'clean' in x_ite_ed_bio_path.name:
        subject_id = int(Path(subject_path).name.split('.')[0].split('S')[1].split('_')[0])
    else:
        subject_id = int(Path(subject_path).stem.split('S')[-1])

    subject_file = Path(subject_path).name
    reasons_list = []
    unique_labels = set(np.unique(subject_labels))

    # compute start indices
    stimuli_start_indices = compute_start_indices(subject_labels, shift=0)

    # check if subject contains all necessary labels
    if not set(accepted_labels) <= unique_labels:
        reason = 'Subject {} is useless. Reason: Missing labels! Expected {} but was {}'.format(subject_id, accepted_labels, unique_labels)
        print(reason)
        reasons_list.append(reason)

    # stays zero, iff a subject does not have any stimuli, which will not be the case
    # since the error would occur earlier
    current_label = 0
    for i, idx in enumerate(stimuli_start_indices):
        current_label = subject_labels[idx]

        # ensure previous label exists
        if i == 0:
            # measurement begins with label != 0
            if current_label != 0:
                reason = 'Subject {} is useless. Reason:' \
                         'Measurement begins with stimulus! Stimulus: {}'.format(subject_id, current_label)
                print(reason)
                reasons_list.append(reason)
        else:
            prev_label = subject_labels[stimuli_start_indices[i - 1]]
            # if previous label is a stimulus and current label is a stimulus, reject subject
            if current_label != 0 and prev_label != 0 and not \
                    (current_label == -10 or current_label == -11 or prev_label == -10 or prev_label == -11):
                reason = 'Subject {} is useless. Reason: No baseline between stimuli! Stimuli {} follows {}'.format(
                    subject_id, current_label, prev_label)
                print(reason)
                reasons_list.append(reason)

            label_length = calculate_label_length(i, idx, stimuli_start_indices, subject_labels)

            # check for baseline
            if current_label == 0:
                if np.abs(prev_label) in [1, 2, 3]:  # phasic baseline
                    if label_length < phasic_baseline_length_tolerance:
                        reason = 'Subject {} is useless. Reason: Phasic baseline too short!' \
                                 'Expected {} but was {}. Prev. label: {}'.format(subject_id,
                                                                                  phasic_baseline_length_tolerance,
                                                                                  label_length, prev_label)
                        print(reason)
                        reasons_list.append(reason)

                if np.abs(prev_label) in [4, 5, 6]:  # tonic baseline
                    if label_length < tonic_baseline_length_tolerance:
                        reason = 'Subject {} is useless. Reason: Tonic baseline too short!' \
                                 'Expected {} but was {}. Prev. label: {}'.format(subject_id,
                                                                                  tonic_baseline_length_tolerance,
                                                                                  label_length, prev_label)
                        print(reason)
                        reasons_list.append(reason)

            elif np.abs(current_label) in [1, 2, 3]:
                if label_length < phasic_length_tolerance:
                    reason = 'Subject {} is useless. Reason: Phasic stimulus {} too short!' \
                             'Expected {} but was {}'.format(subject_id, current_label, phasic_length_tolerance,
                                                             label_length)
                    print(reason)
                    reasons_list.append(reason)
            elif np.abs(current_label) in [4, 5, 6]:
                if label_length < tonic_length_tolerance:
                    reason = 'Subject {} is useless. Reason: Tonic stimulus {} too short!' \
                             'Expected {} but was {}'.format(subject_id, current_label, tonic_length_tolerance,
                                                             label_length)
                    print(reason)
                    reasons_list.append(reason)

    if current_label != 0:
        reason = 'Subject {} is useless. Reason: Measurement ends with stimulus! Stimulus: {}'.format(subject_id,
                                                                                                      current_label)
        print(reason)
        reasons_list.append(reason)

    if len(reasons_list) > 0:
        rejected_subjects.append((subject_file, reasons_list))

print('Purge completed! Excluded {} subjects!'.format(len(rejected_subjects)))

with open(result_file_path, "w") as fd:
    fd.write("\n".join(str(i) for i in rejected_subjects))
