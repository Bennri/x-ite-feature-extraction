import pandas as pd
import scipy.io
from pathlib import Path
from collections import Counter
from feature_extraction.helpers import compute_start_indices
from pprint import pprint
from datetime import datetime
import sys

"""
Count stimuli by intensity per subject and store it in a CSV file.
"""

# TODO set path to dataset
x_ite_ed_bio_path = Path("")

# load file list
file_list = list(x_ite_ed_bio_path.glob('**/*.mat'))
subjects_file_list = file_list
# sort file list
if 'clean' in x_ite_ed_bio_path.name:
    subjects_file_list.sort(key=lambda x: int(x.name.split('.')[0].split('S')[1].split('_')[0]))
    csv_file_name = 'n_stimuli_per_subj_clean.csv'
else:
    subjects_file_list.sort(key=lambda x: int(x.name.split('.')[0].split('S')[1]))
    csv_file_name = 'n_stimuli_per_subj.csv'

# save all counts per subject as dictionaries in a list
subj_counter_list = []
start = datetime.now()
print('Start counting ... ')
for subj_path in subjects_file_list:
    print('\rCurrent subject: {}'.format(subj_path.name), end="")
    sys.stdout.flush()
    subj = scipy.io.loadmat(subj_path)
    subj_labels = subj['stimuli']

    subj_label_counts = {
        'id': '',
        0: 0,
        1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0,
        -10: 0,
        -11: 0, -1: 0, -2: 0, -3: 0, -4: 0, -5: 0, -6: 0,
    }

    # compute start indices to get the labels of each sequence
    subj_start_idx = compute_start_indices(subj_labels)
    # get the corresponding labels and count those
    # label as key, occurrences of label as value
    n_stimuli = Counter(list(subj_labels[subj_start_idx].reshape(-1, )))
    # get the id of the current subject
    if 'clean' in x_ite_ed_bio_path.name:
        subject_id = int(Path(subj_path).stem.split('S')[-1].split('_')[0])
    else:
        subject_id = int(Path(subj_path).stem.split('S')[-1])
    subj_label_counts['id'] = subject_id
    # based on the occurring labels, set the counts (if no label -10 or -11 is present, the pre-set count
    # is untouched
    for k in n_stimuli.keys():
        subj_label_counts[k] = n_stimuli[k]
    # append the dictionary to the list of all stimuli counts per subject
    subj_counter_list.append(subj_label_counts)

end = datetime.now()
print()
print('Counting done')
print('Duration: {}'.format(end - start))
print('Writing data frame.')
# create a data frame from the list of dictionaries
df = pd.DataFrame(subj_counter_list)
pprint(df.head())
df.to_csv(csv_file_name)
