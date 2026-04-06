# Uncertainty-aware-deep-learning-model-for-fatigue-regression-
Uncertainty-aware deep learning model for fatigue regression  using surface electromyography
This repository presents a deep learning framework for predicting Remaining Endurance Time (RET) from surface EMG signals. The approach combines convolutional neural networks (CNN) for spatial–spectral feature extraction, long short-term memory (LSTM) networks for temporal modeling, and Monte Carlo (MC) dropout for predictive uncertainty estimation. 

data/: Dataset (CWT representations and RET labels) 
(Preprocessing.m, compute_SQI_full_v3.m, preprocessing_emg_research.m, windowing_RET_label.m, window_RET_dataset_builder.m, CWT_pipeline.m)
models/ : Saved trained models
cnn_baseline.m: CNN baseline model,
cnn_lstm.m : CNN + LSTM model
CNN_LSTM_MCdropout.m and _f.m: Uncertainty estimation
