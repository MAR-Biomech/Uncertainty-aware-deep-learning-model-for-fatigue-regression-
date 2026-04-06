function DATA = window_RET_dataset_builder(emg_clean, SQI, Fs, win_sec, overlap, plotFlag)

emg_clean = emg_clean(:);
SQI = SQI(:);

N = length(emg_clean);
Tf = N/Fs;

fprintf('\n===== WINDOW + RET DATASET BUILDING =====\n');
fprintf('Total Duration = %.2f sec\n', Tf);

win = round(win_sec*Fs);
step = round(win*(1-overlap));

fprintf('Window Size = %d samples (%.2f sec)\n', win, win_sec);
fprintf('Overlap = %.2f\n', overlap);
fprintf('Step Size = %d samples\n', step);

X = {};
RET = [];
SQI_win = [];
T_win = [];

count = 1;

for i = 1:step:(N-win)

    seg = emg_clean(i:i+win-1);
    sqi_seg = mean(SQI(i:i+win-1));

    t_now = (i + win/2)/Fs;   % center-based labeling
    ret_now = Tf - t_now;

    X{count} = seg;
    RET(count,1) = ret_now;
    SQI_win(count,1) = sqi_seg;
    T_win(count,1) = t_now;

    count = count + 1;

end

DATA.X = X;
DATA.RET = RET;
DATA.SQI = SQI_win;
DATA.T = T_win;
DATA.Tf = Tf;

fprintf('Total Windows Created = %d\n', length(RET));
fprintf('RET Range: %.2f sec to %.2f sec\n', min(RET), max(RET));
fprintf('Mean RET = %.2f sec\n', mean(RET));
fprintf('SQI Range (Windowed): %.4f to %.4f\n', min(SQI_win), max(SQI_win));
fprintf('===== DATASET BUILDING FINISHED =====\n');

end