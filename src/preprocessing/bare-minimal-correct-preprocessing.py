import numpy as np
from scipy import signal

def apply_highpass_filter(ecg, sampling_rate):
    b, a = signal.butter(4, 0.5 / (sampling_rate / 2), 'high')
    return signal.filtfilt(b, a, ecg)

def apply_60hz_notch_filter(ecg, sampling_rate):
    if sampling_rate > 120:
        b, a = signal.iirnotch(60 / (sampling_rate / 2), 40)
        return signal.filtfilt(b, a, ecg)
    else:
        return ecg

def normalize_with_robust_zscore(ecg):
    median_val = np.median(ecg)
    mad = np.median(np.abs(ecg - median_val))
    return (ecg - median_val) / (1.4826 * mad) / 50.0

def resample_to_target_frequency(ecg, sampling_rate):
    if sampling_rate != 256:
        return signal.resample_poly(ecg, 256, sampling_rate)
    else:
        return ecg

def preprocess_ecg_minimal(ecg, sampling_rate):
    """Minimal preprocessing to match pretrained model"""
    # Cut to 30-sec epochs
    n_epochs = len(ecg) // (sampling_rate * 30)
    ecg = ecg[:n_epochs * sampling_rate * 30]
    
    ecg = apply_highpass_filter(ecg, sampling_rate)
    ecg = apply_60hz_notch_filter(ecg, sampling_rate)
    ecg = resample_to_target_frequency(ecg, sampling_rate)
    ecg = normalize_with_robust_zscore(ecg)
    sampling_rate = 256
    
    # Clip and reshape
    ecg = np.clip(ecg, -1, 1).reshape(n_epochs, 7680)
    return ecg


ecg_data = np.loadtxt('test_ecg_data.txt')
epochs = preprocess_ecg_minimal(ecg_data, 256)