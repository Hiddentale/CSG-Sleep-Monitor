import numpy as np
from scipy import signal

def apply_chebyshev_highpass_filter(ecg, sampling_rate):
    passband_freq = 0.5
    stopband_freq = 0.25
    passband_ripple = 0.1
    stopband_attenuation = 60
    
    sos = signal.cheby2(N=4, rs=stopband_attenuation, 
                       Wn=passband_freq/(sampling_rate/2), 
                       btype='high', output='sos')
    
    return signal.sosfiltfilt(sos, ecg)

def apply_60hz_notch_filter(ecg, sampling_rate):
    if sampling_rate > 120:
        center_freq = 60
        bandwidth = 1.5
        Q = center_freq / bandwidth
        
        b, a = signal.iirnotch(center_freq/(sampling_rate/2), Q)
        return signal.filtfilt(b, a, ecg)
    else:
        if sampling_rate >= 70:
            alias_freq = sampling_rate - 60
            Q = alias_freq / 1.5
            b, a = signal.iirnotch(alias_freq/(sampling_rate/2), Q)
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
    
    ecg = apply_chebyshev_highpass_filter(ecg, sampling_rate)
    ecg = apply_60hz_notch_filter(ecg, sampling_rate)
    ecg = resample_to_target_frequency(ecg, sampling_rate)
    sampling_rate = 256
    ecg = normalize_with_robust_zscore(ecg)

    # Clip and reshape
    ecg = np.clip(ecg, -1, 1).reshape(n_epochs, 7680)
    return ecg


ecg_data = np.loadtxt('test_ecg_data.txt')
epochs = preprocess_ecg_minimal(ecg_data, 256)