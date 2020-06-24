import numpy as np
from hrv import HRV
from scipy.stats import iqr
from scipy.stats import median_absolute_deviation
from scipy.signal import argrelextrema
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import normalize
from scipy.signal import stft
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from scipy.signal import coherence
from scipy.stats import pearsonr
from scipy.stats import gaussian_kde
from scipy.stats import skew
from scipy.stats import kurtosis
from ecgdetectors import Detectors


def compute_derivative(signal, h=1):
    # forward difference
    return np.array([np.divide(signal[i + h] - signal[i], h) for i in range(signal.shape[0] - 1 - h)])


# root mean square value
def signal_rms(signal):
    return np.sqrt(np.mean(np.square(signal)))


# mean of local maximum values
def signal_mean_local_max(signal):
    local_max = argrelextrema(signal, np.greater)
    return np.mean(signal[local_max])


# mean of local minimum values
def signal_mean_local_min(signal):
    local_min = argrelextrema(signal, np.less)
    return np.mean(signal[local_min])


# mean of absolute values (mav) of the signal
def mean_absolute_values(signal):
    return np.mean(np.abs(signal))


# zero crossing counts
def signal_zero_crossing(signal):
    return np.size(np.where(np.diff(np.sign(signal), axis=0))[0])


# paper Cao, Cheng and Slobounov, S. (2011). Application of a novel measure of EEG non-stationarity
# as ‘Shannon-entropy of the peak frequency shifting’for detecting residual abnormalities in concussed individuals.
# Clinical Neurophysiology, 122, 1314–1321.
# equation (1) if this definition matches the function
def signal_split_equal_part_mean(signal, sample_rate=1000, factor=10):
    n = len(signal)
    # a proposal to get an almost equal split with an appropriate size of each sub array
    # same is done in the function signal_split_equal_part_std
    # if we have a tonic stimuli which normally has a length of e.g. 60000 samples, then
    # approx. 60 sub arrays are created
    if n < sample_rate * factor:
        t_i = n / sample_rate
    else:
        # to get also an almost equal number of sub arrays for phasic stimuli, the number of samples per sub array
        # is adjusted by the factor, which leads to a 10th factor less samples per sub array
        t_i = n / sample_rate * factor
    split_array = np.array_split(signal, t_i)
    mean_values_partitions = np.array([np.mean(x_i) for x_i in split_array])
    return np.std(mean_values_partitions)


# paper Cao, Cheng and Slobounov, S. (2011). Application of a novel measure of EEG non-stationarity as
# ‘Shannon-entropy of the peak frequency shifting’for detecting residual abnormalities in concussed individuals.
# Clinical Neurophysiology, 122, 1314–1321.
# but this time equation (2)
def signal_split_equal_part_std(signal, sample_rate=1000, factor=10):
    n = len(signal)
    # explanation see function signal_split_equal_part_mean
    if n < sample_rate * factor:
        t_i = n / sample_rate
    else:
        t_i = n / sample_rate * factor
    split_array = np.array_split(signal, t_i)
    std_values_partitions = np.array([np.std(x_i) for x_i in split_array])
    return np.std(std_values_partitions)


# Variance according to Cao, Cheng and Slobounov, S. (2011).
# Application of a novel measure of EEG non-stationarity as ‘Shannon-entropy of the peak frequency shifting’for
# detecting residual abnormalities in concussed individuals. Clinical Neurophysiology, 122, 1314–1321.
def signal_split_equal_part_var(signal, sample_rate=1000, factor=10):
    n = len(signal)
    # explanation see function signal_split_equal_part_mean
    if n < sample_rate * factor:
        t_i = n / sample_rate
    else:
        t_i = n / sample_rate * factor
    split_array = np.array_split(signal, t_i)
    std_values_partitions = np.array([np.std(x_i) for x_i in split_array])
    return np.var(std_values_partitions) # equation (2) in the paper


# variance
def signal_var(signal):
    return np.var(signal)


def mean_rr(signal, sample_rate=1000):
    detectors = Detectors(sample_rate)
    # P.S. Hamilton, “Open Source ECG Analysis Software Documentation”, E.P.Limited, 2002
    r_peaks = detectors.hamilton_detector(signal)
    return np.mean(np.diff(r_peaks))


def mean_hr(ecg_signal, sample_rate=1000):
    hr_analyzer = HRV(sample_rate)
    detectors = Detectors(sample_rate)
    r_peaks = detectors.hamilton_detector(ecg_signal)
    heart_r = hr_analyzer.HR(r_peaks)
    return np.mean(heart_r)


def rms_diffs(ecg_signal, sample_rate=1000):
    detectors = Detectors(sample_rate)
    r_peaks = detectors.hamilton_detector(ecg_signal)
    return np.sqrt(np.divide(np.sum(np.power(np.diff(r_peaks), 2)), (len(r_peaks) - 1)))


def std_absolute_values(signal):
    return np.std(np.abs(signal))


def rms_of_successive_diffs(ecg_signal, sample_rate=1000):
    hr_analyzer = HRV(sample_rate)
    detectors = Detectors(sample_rate)
    r_peaks = detectors.hamilton_detector(ecg_signal)
    return hr_analyzer.RMSSD(r_peaks, normalise = False)


def std_of_successive_diffs(ecg_signal, sample_rate=1000):
    hr_analyzer = HRV(sample_rate)
    detectors = Detectors(sample_rate)
    r_peaks = detectors.hamilton_detector(ecg_signal)
    return hr_analyzer.SDSD(r_peaks)


def mean_value_first_diff(signal, h=1):
    return np.mean(compute_derivative(signal, h))


def mean_value_second_diff(signal, h=2):
    return np.mean(compute_derivative(signal, h))


def max_to_min_peak_value_ratio(signal):
    return np.divide(np.max(signal), np.min(signal))


"""
some features according to:
Werner, P., Al-Hamadi, A., Limbrecht-Ecklundt, K., Walter, S., Gruss, S., & Traue, H. C. (2017).
Automatic Pain Assessment with Facial Activity Descriptors. IEEE Transactions on Affective Computing, 8(3),
286–299. https://doi.org/10.1109/TAFFC.2016.2537327
"""


# mean absolute value of sample x_i and x_{i+1} (first differences) also called mavfd by Werner et al.
# in Automatic Pain Recognition from Video and Biomedical Signals
def mean_absolute_value_first_diff(signal, h=1):
    # return np.mean(np.array([np.abs(signal[i + 1] - signal[i]) for i in range(signal.shape[0] - 1)]))
    return np.mean(np.abs(compute_derivative(signal, h)))


# mean absolute value of sample x_i and x_{i+2}
def mean_absolute_value_second_diff(signal, h=2):
    # return np.mean(np.array([np.abs(signal[i + 2] - signal[i]) for i in range(signal.shape[0] - 2)]))
    return np.mean(np.abs(compute_derivative(signal, h)))


def signal_p2pmv(signal):
    local_max = argrelextrema(signal, np.greater)
    local_min = argrelextrema(signal, np.less)
    return np.mean(local_max) - np.mean(local_min)


def signal_mean(signal):
    return np.mean(signal)


def signal_median(signal):
    return np.median(signal)


def signal_min(signal):
    return np.min(signal)


def signal_max(signal):
    return np.max(signal)


def signal_range(signal):
    return np.max(signal) - np.min(signal)


def signal_std(signal):
    return np.std(signal)


# inter-quartile range
def signal_iqr(signal):
    return iqr(signal)


# inter-decile range
def signal_idr(signal):
    return np.percentile(signal, 90) - np.percentile(signal, 10)


def signal_mad(signal):
    return median_absolute_deviation(signal)


# instant of time of signal's maximum value
def signal_tmax(signal, sample_rate=1000):
    return np.argmax(signal) / sample_rate


# number of samples the signal is greater than mean -> duration
def signal_tgm(signal, sample_rate=1000):
    signal_mean = np.mean(signal)
    return np.size(signal[signal > signal_mean]) / sample_rate


# how long does the signal have a higher value than the average of mean and min of the signal?
def signal_tga(signal, sample_rate=1000):
    s_mean = np.mean(signal)
    s_min = np.min(signal)
    avg_mean_and_min = np.divide(s_mean + s_min, 2)
    return np.size(signal[signal > avg_mean_and_min]) / sample_rate


# number of segments signal > mean
def signal_sgm(signal):
    signal_mean = np.mean(signal)
    segments = 0
    above_mean = False
    for x in signal:
        if x >= signal_mean and not above_mean:
            above_mean = True
            segments += 1
        elif x < signal_mean and above_mean:
            above_mean = False
    return segments


# number of segments signal > (mean + min) / 2
def signal_sga(signal):
    s_mean = np.mean(signal)
    s_min = np.min(signal)
    avg_mean_and_min = np.divide(s_mean + s_min, 2)
    segments = 0
    above_mean = False
    for x in signal:
        if x >= avg_mean_and_min and not above_mean:
            above_mean = True
            segments += 1
        elif x < avg_mean_and_min and above_mean:
            above_mean = False
    return segments


def signal_area(signal, x=None, dx=1):
    area = np.trapz(signal, x=x, dx=dx)
    return area


# areaR
def signal_area_min_max(signal, x=None, dx=1):
    s_min = np.argmin(signal)
    s_max = np.argmax(signal)

    if s_min < s_max:
        signal_range = signal[s_min:s_max]
    else:
        signal_range = signal[s_max:s_min]

    area_min_max = np.trapz(signal_range, x, dx)
    if np.isclose(area_min_max, 0):
        return 0.0
    return area_min_max


def signal_area_min_max_ratio(signal, x=None, dx=1):
    area = signal_area(signal, x, dx)
    area_min_max = signal_area_min_max(signal, x, dx)
    return area_min_max / area


def signal_first_derivative(signal, func_list=None):
    if func_list is None:
        func_list = []
    first_deriv = compute_derivative(signal)
    feature_list = []
    for f in func_list:
        feature_list.append(f(first_deriv))
    return feature_list


def signal_second_derivative(signal, func_list=None):
    if func_list is None:
        func_list = []
    second_deriv = compute_derivative(compute_derivative(signal))
    feature_list = []
    for f in func_list:
        feature_list.append(f(second_deriv))
    return feature_list
