"""
ECG preprocessing module for sleep stage classification.

Based on the methodology from Jones et al. (2024).
"""

from .ecg_preprocessing import preprocess_ecg_pipeline
from .signal_filters import apply_highpass_filter, remove_powerline_noise
from .heartbeat_detection import detect_heartbeats_template_matching
from .quality_validation import validate_recording_quality

__all__ = [
    'preprocess_ecg_pipeline',
    'apply_highpass_filter',
    'remove_powerline_noise',
    'detect_heartbeats_template_matching',
    'validate_recording_quality',
]