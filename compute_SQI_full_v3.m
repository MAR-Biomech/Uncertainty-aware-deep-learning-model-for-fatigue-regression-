function SQI = compute_SQI_full_v3(emg, Fs, plotFlag)

emg = emg(:);
N = length(emg);

fprintf('\n===== SQI COMPUTATION STARTED =====\n');
fprintf('Signal Length = %d samples\n', N);
fprintf('Sampling Frequency = %.2f Hz\n', Fs);

win_sec = 0.5;
win = round(win_sec*Fs);
step = round(win/2);

fprintf('Window Size = %d samples\n', win);
fprintf('Step Size = %d samples\n', step);

SNR_all = [];
Band_all = [];
Env_all = [];
idx_all = [];

k = 1;

for i = 1:step:(N-win)

    seg = emg(i:i+win-1);

    signal_power = mean(seg.^2);
    hp = highpass(seg,20,Fs);
    noise_power = var(seg-hp);
    SNR = log(signal_power/(noise_power+eps) + 1);

    [Pxx,f] = pwelch(seg,[],[],[],Fs);
    band_pow = bandpower(Pxx,f,[20 450],'psd');
    total_pow = bandpower(Pxx,f,'psd');
    Band = band_pow/(total_pow+eps);

    env = abs(hilbert(seg));
    Env = 1/(var(env)+eps);

    SNR_all(k) = SNR;
    Band_all(k) = Band;
    Env_all(k) = Env;
    idx_all(k) = i;

    k = k + 1;

end

fprintf('Total Windows Processed = %d\n', length(SNR_all));

SNR_all = rescale(SNR_all);
Band_all = rescale(Band_all);
Env_all = rescale(Env_all);

SQI_win = 0.4*SNR_all + 0.3*Band_all + 0.3*Env_all;

SQI = zeros(N,1);
count = zeros(N,1);

for k = 1:length(SQI_win)
    i = idx_all(k);
    SQI(i:i+win-1) = SQI(i:i+win-1) + SQI_win(k);
    count(i:i+win-1) = count(i:i+win-1) + 1;
end

SQI = SQI ./ (count + eps);
SQI = fillmissing(SQI,'nearest');

fprintf('SQI Range: %.4f to %.4f\n', min(SQI), max(SQI));
fprintf('Mean SQI = %.4f\n', mean(SQI));
fprintf('===== SQI COMPUTATION FINISHED =====\n');

end