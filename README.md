This dataset presents a comprehensive deep learning framework for estimating Remaining Endurance Time (RET) from surface electromyography (sEMG) signals. The proposed approach integrates spatial–spectral feature extraction, temporal sequence modeling, and uncertainty quantification to provide both accurate and reliable fatigue predictions.

The pipeline begins by transforming raw EMG signals into time–frequency representations using Continuous Wavelet Transform (CWT). These representations are then processed using a Convolutional Neural Network (CNN) to extract discriminative spatial–spectral features associated with fatigue-related changes. To capture the temporal evolution of fatigue, the extracted features are passed through a Long Short-Term Memory (LSTM) network, enabling the model to learn sequential dependencies and represent fatigue as a continuous process rather than a static state.

In addition to prediction accuracy, this work incorporates uncertainty estimation using Monte Carlo (MC) dropout. During inference, multiple stochastic forward passes are performed to obtain a distribution of predictions. The variability of these predictions is used to quantify predictive uncertainty, providing confidence-aware outputs that are particularly important for physiological signals such as EMG, which are inherently noisy and variable.

Key Features:

End-to-end pipeline from EMG → CWT → RET prediction
Comparison of three models: CNN (baseline), CNN + LSTM, and CNN + LSTM with uncertainty 

Data Directory (data/):

Contains dataset preparation and preprocessing scripts:

Preprocessing.m
Main preprocessing script for raw EMG signals
compute_SQI_full_v3.m
Computes signal quality indices (SQI) to assess EMG reliability
preprocessing_emg_research.m
Advanced EMG preprocessing (filtering, normalization, artifact handling)
windowing_RET_label.m
Generates RET labels aligned with EMG segments
window_RET_dataset_builder.m
Constructs dataset samples using sliding window approach
CWT_pipeline.m
Converts EMG signals into CWT-based time–frequency images

These steps ensure robust signal conditioning and proper alignment between EMG data and fatigue labels.

Models Directory (models/):

Contains trained models and implementation scripts:

cnn_baseline.m
CNN-based regression model for RET prediction (no temporal modeling)
cnn_lstm.m
CNN + LSTM model incorporating temporal dependencies
CNN_LSTM_MCdropout.m
CNN–LSTM model with Monte Carlo dropout for uncertainty estimation
CNN_LSTM_MCdropout_f.m
Extended version for uncertainty-aware modeling in structured physiological (respiratory-related) analysis
