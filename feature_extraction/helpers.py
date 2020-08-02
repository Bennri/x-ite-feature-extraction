import numpy as np
# from .feature_extraction_functions import signal_stationarity_per_freq


def apply_on_channel(data, apply_to_channel, applied_func, default_channel_names=None):
    """
    Applies a function on a specific channel within the data set.
    :param data: A matrix with shape (samples, channels).
    :param apply_to_channel: The channel on which the function will be applied, provided as a string.
    :param applied_func: The function to apply.
    :param default_channel_names: The resolution of channel names to integer values / columns in the data matrix.
    :return: The result of the given function.
    """
    if default_channel_names is None:
        default_channel_names = {'corrugator': 0, 'zygomaticus': 1, 'trapezius': 2, 'scl': 3, 'ecg': 4}
    int_channel = default_channel_names[apply_to_channel]
    result_channel = applied_func(data[:, int_channel])
    return result_channel


def process_dataset_preprocessing(dataset, channel_filter_dict, default_channel_names=None):
    """
    Applies functions such as filtering on a channel of the given data matrix.
    :param dataset: A matrix with shape (sample, channel).
    :param channel_filter_dict: A dictionary which holds for each channel name a list of preprocessing functions.
    :param default_channel_names: The resolution from channel name to column.
    :return: The modified / filtered data set.
    """
    if default_channel_names is None:
        default_channel_names = {'corrugator': 0, 'zygomaticus': 1, 'trapezius': 2, 'scl': 3, 'ecg': 4}
    for channel_key in channel_filter_dict.keys():
        for i, func in enumerate(channel_filter_dict[channel_key]):
            dataset[:, default_channel_names[channel_key]] = apply_on_channel(data=dataset,
                                                                              apply_to_channel=channel_key,
                                                                              applied_func=func,
                                                                              default_channel_names=default_channel_names).reshape(-1,)
    return dataset


def process_slice_preprocessing(dataset, channel_filter_dict, default_channel_names=None):
    """
    Applies functions such as filtering on a channel of the given data slice matrix.
    This function is needed since the slices are processed in parallel and each process has read only access on the copied
    slice. Therefore, a new array is created.
    :param dataset: A data slice with shape (sample, channel).
    :param channel_filter_dict: A dictionary which holds for each channel name a list of preprocessing functions.
    :param default_channel_names: The resolution from channel name to column.
    :return: The modified / filtered data set.
    """

    pre_processed_slice = np.zeros(dataset.shape)
    if default_channel_names is None:
        default_channel_names = {'corrugator': 0, 'zygomaticus': 1, 'trapezius': 2, 'scl': 3, 'ecg': 4}
    for channel_key in channel_filter_dict.keys():
        for i, func in enumerate(channel_filter_dict[channel_key]):
            pre_processed_slice[:, default_channel_names[channel_key]] = apply_on_channel(data=dataset,
                                                                                          apply_to_channel=channel_key,
                                                                                          applied_func=func,
                                                                                          default_channel_names=default_channel_names).reshape(
                -1, )
    return dataset


def compute_start_indices(stimuli_labels, shift=0):
    """
    Computes all indices at which a new label different to the one before occurs. That demonstrates when a new stimuli sequence starts.
    :param stimuli_labels: All labels as numpy array (of a subject).
    :param shift: A wanted shift from the actual start.
    :return: A numpy array of indices at which a new stimuli starts.
    """
    current_label = stimuli_labels[0 + shift]
    start_indices = [0 + shift]

    for i, label in enumerate(stimuli_labels):

        if label == current_label:
            continue
        else:
            if i + shift >= stimuli_labels.shape[0]:
                return np.array(start_indices)
            else:
                start_indices.append(i + shift)
                current_label = label

    return np.array(start_indices)


def process_dataset_extract_features(stimuli_channel_matrix, channel_features_dict,
                                     default_order=None, default_channel_names=None):
    """

    :param stimuli_channel_matrix:
    :param channel_features_dict:
    :param default_order:
    :param default_channel_names:
    :return:
    """
    # to have the same order of processing and therefore within the feature vector when the function is called
    if default_order is None:
        default_order = ['corrugator', 'zygomaticus', 'trapezius', 'scl', 'ecg']
    elif set(channel_features_dict.keys()) != set(default_order) or len(list(channel_features_dict.keys())) != len(
            default_order):
        default_order = ['corrugator', 'zygomaticus', 'trapezius', 'scl', 'ecg']

    if default_channel_names is None:
        default_channel_names = {'corrugator': 0, 'zygomaticus': 1, 'trapezius': 2, 'scl': 3, 'ecg': 4}

    # since number of features differ per channel and therefore differ in index shifts a numpy array
    # cannot be used in the beginning
    # instead collect the features per channel in a list and flat / stack the list at the end to get the
    # whole feature vector which then will be returned
    accumulator_features = []
    for channel_key in default_order:
        # current amount of features based on the channel and the length of the list of feature functions
        # current_features = np.zeros(len(channel_features_dict[channel_key]))
        current_features = []
        for i, func in enumerate(channel_features_dict[channel_key]):
            features = apply_on_channel(data=stimuli_channel_matrix,
                                        apply_to_channel=channel_key,
                                        applied_func=func,
                                        default_channel_names=default_channel_names)
            # in case of that a function returns multiple values which then has to be a list
            if type(features) == list:
                current_features = current_features + features
            # or a numpy array
            elif type(features) == np.ndarray:
                current_features = current_features + list(features)
            # or a single value
            else:
                current_features.append(features)
        # append the feature array of that channel to the list
        accumulator_features.append(current_features)
    # flat / horizontal stack the list to get a feature vector as numpy array
    feature_vec = np.hstack(accumulator_features)
    return feature_vec


# def extract_freq_domain_features(signal, func_list=None, sample_rate=1000):
#     """
#     Computes the frequency components of the given signal by utilizing FFT. Afterwards applies a list of functions
#     to the resulting components, all within the frequency domain.
#     :param signal: The given signal from which all frequency domain features shall be computed.
#     :param func_list: The list of functions from which the features wihtin the frequency domain will be computed.
#     :param sample_rate: The sample rate of the singal, default 1000.
#     :return: A python list containing all extracted features from the frequency domain.
#     """
#     if func_list is None:
#         func_list = []
#     # list in which all features will be stored
#     features_freq_domain = []
#     # compute the frequency components and do a reshape from (n_samples, 1) to (n_samples, )
#     # since otherwise 2D-arrays have to be handled
#     freq_components = np.fft.fft(signal.reshape(-1, ))
#     # iterate over all functions
#     for func in func_list:
#         # print('Applying function: {}'.format(func.__str__()))
#         # apply a function
#         feature = func(freq_components, sample_rate)
#         # determine return type of that function and store the feature
#         if type(feature) == tuple:
#             features_freq_domain = features_freq_domain + list(feature)
#         elif type(feature) == list:
#             features_freq_domain = features_freq_domain + feature
#         else:
#             features_freq_domain.append(feature)
#
#     return features_freq_domain


# def extract_stationarity_features(signal, func_list=None, sample_rate=1000):
#     """
#     Computes the frequency components of the given signal by utilizing FFT. Afterwards applies a list of functions
#     to the resulting components, all within the frequency domain.
#     :param signal: The given signal from which all frequency domain features shall be computed.
#     :param func_list: The list of functions from which the features wihtin the frequency domain will be computed.
#     :param sample_rate: The sample rate of the singal, default 1000.
#     :return: A python list containing all extracted features from the frequency domain.
#     """
#     if func_list is None:
#         func_list = []
#     # list in which all features will be stored
#     features_stationarity = []
#     # compute the frequency components and do a reshape from (n_samples, 1) to (n_samples, )
#     # since otherwise 2D-arrays have to be handled
#     _, freq_stationarity = signal_stationarity_per_freq(signal, sample_rate)
#     # iterate over all functions
#     for func in func_list:
#         # apply a function
#         feature = func(freq_stationarity)
#         # determine return type of that function and store the feature
#         if type(feature) == tuple:
#             features_stationarity = features_stationarity + list(feature)
#         elif type(feature) == list:
#             features_stationarity = features_stationarity + feature
#         else:
#             features_stationarity.append(feature)
#
#     return features_stationarity


def replace_or_remove_occurring_nans_in_data_set(data, strategy='mean', filter_function=np.isnan):
    """
    Replaces the occurring NaN values (if any) with a numeric value computed based on a given strategy.
    Strategies are computing the mean value from a subject'c column and set that value. Another strategy is to set zero
    instead of NaN. Or remove the feature vectors which as NaN values in it.
    :param data: The data matrix in which the NaN values shall be replaced.
    :param strategy: Mean, zero, remove as a strategy key word for replacing the NaN values. Defaults to 'mean'.
    :param filter_function: A function to find positions of the values which shall be replaced with the given strategy.
    Default: numpy.isnan, but can also be used e.g. with numpy.isinf
    :return: The data matrix without NaN values, filled with the given strategy.
    """
    print('Data shape before filtering {}: {}'.format(filter_function.__name__, data.shape))
    nan_row, nan_col = np.where(filter_function(data))
    rows_to_delete = []  # remember rows, where mean is not possible (e.g. ill tonic stimuli)
    if strategy == 'remove':
        data = np.delete(data, nan_row, axis=0)
        print("Deleted rows: {}".format(nan_row))
    else:
        for i, row in enumerate(nan_row):
            # get subject ID from that row
            subj_id = data[row, 0]
            affected_label = data[row, -1]
            # filter data from that subject and the equal label as the one of that row
            if strategy == 'mean':
                data_only_from_subj = data[np.where(data[:, 0] == subj_id)[0], :]
                # get only the samples of subject with the same label
                data_only_from_subj = data_only_from_subj[np.where(data_only_from_subj[:, -1] == affected_label)]
                # compute mean from the resulting filtered data and fill the nan values with the mean
                data_col = data_only_from_subj[np.logical_and(~np.isnan(data_only_from_subj[:, nan_col[i]]),
                                                              ~np.isinf(data_only_from_subj[:, nan_col[i]]))][:, nan_col[i]]
                if len(data_col) == 0:
                    print("Feature vector deletion for subject {}, label {}, feature: {}".format(subj_id,
                                                                                                 affected_label,
                                                                                                 nan_col[i]))
                    rows_to_delete.append(row)
                else:
                    data[row, nan_col[i]] = np.mean(data_col)
            elif strategy == 'zero':
                data[row, nan_col[i]] = 0.0
        data = np.delete(data, rows_to_delete, axis=0)
        print("Repaired cells are in rows: {}, cols: {}".format(nan_row, nan_col))
        print("Deleted rows due to detected {}: {}".format(filter_function.__name__, rows_to_delete))
    print('Data shape after filtering {}: {}'.format(filter_function.__name__, data.shape))
    repaired_dict = {
        'repaired_rows': [int(x) for x in nan_row],
        'repaired_cols': [int(x) for x in nan_col],
        'deleted_rows': [int(x) for x in rows_to_delete],
        'applied_function': filter_function.__name__
    }
    return data, repaired_dict
