clc; clear; close all;

%% SETTINGS
Fs = 1259.26;
filename = 'Fatigue_legextension.xlsx';

win_sec = 1;
overlap = 0.5;

sheets = sheetnames(filename);

muscle_list = {'RF','VM','VL'};

%% STORAGE
ALL_CWT = [];
ALL_RET = [];
ALL_SQI = [];
ALL_SUBJECT = [];
ALL_MUSCLE = [];

global_count = 1;

%% =====================================================
%% LOOP PARTICIPANTS
%% =====================================================

for p = 1:length(sheets)

    fprintf('Participant %d\n',p);

    T = readtable(filename,'Sheet',sheets{p});

    %% =====================================================
    %% LOOP MUSCLES
    %% =====================================================

    for m = 1:length(muscle_list)

        muscle_name = muscle_list{m};

        fprintf('   Muscle %s\n', muscle_name);

        emg = fillmissing(T.(muscle_name),'linear');

% 🔹 Convert Volts → millivolts
emg = emg * 1000;

fprintf('      Converted to mV | Range: %.4f to %.4f mV\n', ...
        min(emg), max(emg));

        %% SQI
        SQI = compute_SQI_full_v3(emg, Fs, 0);

        %% PREPROCESS
        emg_clean = preprocess_emg_research(emg, Fs, SQI, 0);

        %% WINDOW + RET
        WINDOW_DATA = window_RET_dataset_builder(emg_clean, SQI, Fs, win_sec, overlap, 0);

        %% CWT
        for k = 1:length(WINDOW_DATA.RET)

            seg = WINDOW_DATA.X{k};

            img = build_CWT_scalogram(seg, Fs);

            ALL_CWT(:,:,1,global_count) = img;
            ALL_RET(global_count,1) = WINDOW_DATA.RET(k);
            ALL_SQI(global_count,1) = WINDOW_DATA.SQI(k);
            ALL_SUBJECT(global_count,1) = p;
            ALL_MUSCLE(global_count,1) = m;

            global_count = global_count + 1;

        end

    end

end
%%
fprintf('\n===== FINAL DATASET SUMMARY =====\n');
fprintf('Total Samples: %d\n', global_count-1);
fprintf('CWT Size: %d x %d\n', size(ALL_CWT,1), size(ALL_CWT,2));
fprintf('RET Range: %.2f to %.2f sec\n', min(ALL_RET), max(ALL_RET));
fprintf('SQI Range: %.4f to %.4f\n', min(ALL_SQI), max(ALL_SQI));
fprintf('Total Subjects: %d\n', length(unique(ALL_SUBJECT)));
fprintf('Total Muscles: %d\n', length(unique(ALL_MUSCLE)));
fprintf('==================================\n');
%% SAVE FINAL DATA
FINAL_DATA.CWT = ALL_CWT;
FINAL_DATA.RET = ALL_RET;
FINAL_DATA.SQI = ALL_SQI;
FINAL_DATA.SUBJECT = ALL_SUBJECT;
FINAL_DATA.MUSCLE = ALL_MUSCLE;

save('FINAL_MULTI_MUSCLE_RET_CWT.mat','FINAL_DATA','-v7.3');

disp('FINAL MULTI MUSCLE DATASET READY')
%%
function CWT_img = build_CWT_scalogram(x, Fs)

[cfs, ~] = cwt(x, Fs, 'amor');

scalogram = abs(cfs);
scalogram = log(scalogram + eps);

CWT_img = imresize(scalogram,[64 64]);
CWT_img = rescale(CWT_img);

end