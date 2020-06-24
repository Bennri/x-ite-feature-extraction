import inspect


channel_feature_extr_config_dict_time_domain = {
    'corrugator': [
        # your functions used for feature extraction incl. name (func.__name__)
    ],
    'zygomaticus': [
        # your functions used for feature extraction incl. name (func.__name__)
    ],
    'trapezius': [
        # your functions used for feature extraction incl. name (func.__name__)
    ],
    'scl': [
        # your functions used for feature extraction incl. name (func.__name__)
    ],
    'ecg': [
        # your functions used for feature extraction incl. name (func.__name__)
    ]
}


def create_feature_names_as_list(features_per_channel_dict, channels_list_order=None):
    if channels_list_order is None:
        channels_list_order = ['corrugator', 'zygomaticus', 'trapezius', 'scl', 'ecg']
    feature_names = []
    for channel_name in channels_list_order:
        feature_func_list = features_per_channel_dict[channel_name]
        for func in feature_func_list:
            sig = inspect.signature(func)
            if 'func_list' in sig.parameters:
                func_list = sig.parameters['func_list'].default
                for sub_func in func_list:
                    feat_name = channel_name + "_" + func.__name__ + "_" + sub_func.__name__
                    feature_names.append(feat_name)
            else:
                feat_name = channel_name + "_" + func.__name__
                feature_names.append(feat_name)
    return feature_names


# example usage
# each row is a feature vector whereas a column represents a feature of the feature vector
# if __name__ == '__main__':
#     import numpy as np
#     import pandas as pd
#     default_o = ['corrugator', 'zygomaticus', 'trapezius', 'scl', 'ecg']
#     feat_names_list = create_feature_names_as_list(channel_feature_extr_config_dict_time_domain,
#                                                    channels_list_order=default_o)
#     cols = ["subj_id"] + feat_names_list + ["label"]
#     data_set = np.load('../example/dataset/xite_data_set.npy', allow_pickle=False, fix_imports=False)
#     df = pd.DataFrame(data_set, columns=cols)
#     df.to_csv("../example/dataset/xite_data_set.csv")
