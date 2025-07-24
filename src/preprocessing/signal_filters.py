"""
Signal filtering functions for ECG preprocessing.
"""

import numpy as np

def apply_highpass_filter(ecg_signal: np.ndarray, sampling_rate: int, cutoff_hz: float = 0.5) -> np.ndarray:
    """
    4
    Apply high-pass filter to attenuate baseline wander while preserving T-waves.
    
    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling frequency in Hz
        cutoff_hz: High-pass filter cutoff frequency (default: 0.5 Hz)
        
    Returns:
        High-pass filtered ECG signal
    """
    pass


def remove_powerline_noise(ecg_signal: np.ndarray, sampling_rate: int, powerline_hz: float = 60.0) -> np.ndarray:
    """
    5
    Remove 60 Hz powerline noise using a notch filter.
    
    Args:
        ecg_signal: ECG signal array
        sampling_rate: Sampling frequency in Hz
        powerline_hz: Powerline frequency to remove (default: 60.0 Hz)
        
    Returns:
        ECG signal with powerline noise removed
    """
    pass
