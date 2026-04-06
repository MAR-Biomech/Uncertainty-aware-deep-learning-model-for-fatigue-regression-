clc; clear; close all;

Fs = 1259.26;
filename = 'Fatigue_legextension.xlsx';

sheets = sheetnames(filename);

T = readtable(filename,'Sheet',sheets{1});
emg = fillmissing(T.RF,'linear');

% 🔹 Convert Volts → mV
emg = emg * 1000;

fprintf('Signal converted from Volts to mV\n');
fprintf('Amplitude range: %.4f to %.4f mV\n', min(emg), max(emg));

%% STEP 1 — SQI
SQI = compute_SQI_full_v3(emg, Fs, 0);

%% STEP 2 — PREPROCESS
emg_clean = preprocess_emg_research(emg, Fs, SQI, 1);