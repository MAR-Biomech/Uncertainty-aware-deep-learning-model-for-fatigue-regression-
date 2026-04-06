function emg_out = preprocess_emg_research(emg_in, Fs, SQI, plotFlag)

emg = emg_in(:);
t = (0:length(emg)-1)/Fs;

fprintf('\n===== PREPROCESSING STARTED =====\n');

%% 1️⃣ Baseline
emg_detrend = detrend(emg,2);
emg_hp = highpass(emg_detrend,10,Fs);
fprintf('Baseline drift removed.\n');

%% 2️⃣ Motion
med_win = round(0.2*Fs);
emg_med = movmedian(emg_hp, med_win);
residual = emg_hp - emg_med;

thr = 3*std(residual);
residual(abs(residual)>thr) = thr*sign(residual(abs(residual)>thr));
emg_motion_clean = emg_med + residual;

fprintf('Motion artifacts suppressed. Threshold = %.4f\n', thr);

%% 3️⃣ Notch
d = designfilt('bandstopiir',...
    'FilterOrder',4,...
    'HalfPowerFrequency1',49,...
    'HalfPowerFrequency2',51,...
    'SampleRate',Fs);

emg_notch = filtfilt(d, emg_motion_clean);
fprintf('Powerline interference removed (50 Hz).\n');

%% 4️⃣ Bandpass
[b,a] = butter(4,[20 450]/(Fs/2),'bandpass');
emg_bp = filtfilt(b,a,emg_notch);
fprintf('Physiological bandpass applied (20–450 Hz).\n');

%% 5️⃣ SQI-Aware Cleaning
SQI = rescale(SQI);
alpha = 0.6 + 0.4*SQI;
emg_sqi = emg_bp .* alpha;

fprintf('SQI-aware soft cleaning applied.\n');
fprintf('Alpha range: %.4f to %.4f\n', min(alpha), max(alpha));

%% 6️⃣ Normalization
emg_out = (emg_sqi - mean(emg_sqi))/std(emg_sqi);

fprintf('Final normalization done.\n');
fprintf('Output Mean = %.4f | Std = %.4f\n', mean(emg_out), std(emg_out));

fprintf('===== PREPROCESSING FINISHED =====\n');

end