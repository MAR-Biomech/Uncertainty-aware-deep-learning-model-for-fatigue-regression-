clc; clear; close all;

%% SETTINGS
Fs = 1259.26;
win_sec = 1;
overlap = 0.5;

filename = 'Fatigue_legextension.xlsx';
sheets = sheetnames(filename);

%% LOAD
T = readtable(filename,'Sheet',sheets{1});
emg = fillmissing(T.RF,'linear');

%% STEP 1 SQI
SQI = compute_SQI_full_v3(emg, Fs, 0);

%% STEP 2 PREPROCESS
emg_clean = preprocess_emg_research(emg, Fs, SQI, 0);

%% STEP 3 WINDOW + RET
DATA = window_RET_dataset_builder(emg_clean, SQI, Fs, win_sec, overlap, 1);