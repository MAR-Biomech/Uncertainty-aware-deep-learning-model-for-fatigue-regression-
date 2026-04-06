clc; clear; close all;

fprintf('\n====================================================\n');
fprintf('   RELIABILITY & RISK ANALYSIS PIPELINE\n');
fprintf('====================================================\n');

%% =====================================================
%% LOAD MODEL
%% =====================================================

model_file = 'CNN_LSTM_REAL_MC_UNCERTAINTY_MODEL.mat';

if ~isfile(model_file)
    error('Uncertainty model not found. Run training first.');
end

load(model_file);

netCNN      = UNC_MODEL.CNN;
netLSTM     = UNC_MODEL.LSTM;
seq_len     = UNC_MODEL.seq_len;
idx_seq     = UNC_MODEL.idx_seq;
Ntrain_seq  = UNC_MODEL.Ntrain_seq;

fprintf('\nModel loaded successfully.\n');

%% =====================================================
%% LOAD DATA & REBUILD TEST SET (CORRECT SPLIT)
%% =====================================================

load FINAL_MULTI_MUSCLE_RET_CWT.mat

X = FINAL_DATA.CWT;
Y = FINAL_DATA.RET(:);
N = size(X,4);

% Recompute CNN features
features = zeros(N,64);
for i = 1:N
    features(i,:) = activations(netCNN,X(:,:,:,i),'relu_feature','OutputAs','rows');
end

% Rebuild sequences
Xseq = {};
Yseq = [];
count = 1;

for i = 1:(N-seq_len)
    Xseq{count} = features(i:i+seq_len-1,:)';
    Yseq(count,1) = Y(i+seq_len-1);
    count = count + 1;
end

% Correct test set
XTest = Xseq(idx_seq(Ntrain_seq+1:end));
YTest = Yseq(idx_seq(Ntrain_seq+1:end));

fprintf('Test samples reconstructed: %d\n',length(YTest));

%% =====================================================
%% RECOMPUTE MC DROPOUT (SAFE VERSION)
%% =====================================================

lgraph = layerGraph(netLSTM.Layers);
lgraph = removeLayers(lgraph,lgraph.Layers(end).Name);
dlnet = dlnetwork(lgraph);

Nmc = 30;
numTest = numel(XTest);

preds = zeros(numTest,Nmc);

for k = 1:Nmc
    for i = 1:numTest
        dlX = dlarray(single(XTest{i}),'CT');
        y = forward(dlnet,dlX);
        preds(i,k) = extractdata(y);
    end
end

mu_pred = mean(preds,2);

p90 = prctile(preds,90,2);
p10 = prctile(preds,10,2);
interval_width = p90 - p10;

sigma_pred = interval_width ./ (abs(mu_pred)+1e-6);
alpha = mean(abs(mu_pred - YTest)) / mean(sigma_pred);
sigma_pred = alpha * sigma_pred;

fprintf('MC Dropout recomputed successfully.\n');

%% =====================================================
%% BASELINE PERFORMANCE
%% =====================================================

errors = abs(mu_pred - YTest);
errors_sq = (mu_pred - YTest).^2;

baseline_rmse = sqrt(mean(errors_sq));
baseline_mae  = mean(errors);

fprintf('\nBaseline RMSE: %.4f sec\n',baseline_rmse);
fprintf('Baseline MAE : %.4f sec\n',baseline_mae);

%% =====================================================
%% RELIABILITY THRESHOLD ANALYSIS
%% =====================================================

T_range = linspace(min(sigma_pred), max(sigma_pred), 40);

coverage_list = zeros(size(T_range));
rmse_list = zeros(size(T_range));
mae_list = zeros(size(T_range));

Ntotal = length(YTest);

for i = 1:length(T_range)

    idx_accept = sigma_pred <= T_range(i);

    coverage_list(i) = sum(idx_accept) / Ntotal;

    if sum(idx_accept) > 10
        rmse_list(i) = sqrt(mean((mu_pred(idx_accept) - YTest(idx_accept)).^2));
        mae_list(i)  = mean(abs(mu_pred(idx_accept) - YTest(idx_accept)));
    else
        rmse_list(i) = NaN;
        mae_list(i)  = NaN;
    end
end

%% Optimal operating point (≥70% coverage)

min_coverage = 0.70;
valid_idx = coverage_list >= min_coverage;

rmse_valid = rmse_list;
rmse_valid(~valid_idx) = Inf;

[best_rmse, best_idx] = min(rmse_valid);

best_threshold = T_range(best_idx);
best_coverage  = coverage_list(best_idx);

fprintf('\n===== RELIABILITY OPERATING POINT =====\n');
fprintf('Threshold : %.4f\n', best_threshold);
fprintf('Coverage  : %.2f %%\n', best_coverage*100);
fprintf('RMSE      : %.4f sec\n', best_rmse);
fprintf('Improvement over baseline: %.2f %%\n',...
    100*(baseline_rmse-best_rmse)/baseline_rmse);

%% =====================================================
%% PERCENTILE ERROR RELIABILITY ANALYSIS
%% =====================================================

Nbins = 8;
edges = linspace(min(sigma_pred), max(sigma_pred), Nbins+1);

p50 = zeros(Nbins,1);
p75 = zeros(Nbins,1);
p90 = zeros(Nbins,1);
p95 = zeros(Nbins,1);
sigma_mid = zeros(Nbins,1);

for b = 1:Nbins
    
    idx_bin = sigma_pred >= edges(b) & sigma_pred < edges(b+1);
    
    if sum(idx_bin) > 10
        
        err_bin = errors(idx_bin);
        
        p50(b) = prctile(err_bin,50);
        p75(b) = prctile(err_bin,75);
        p90(b) = prctile(err_bin,90);
        p95(b) = prctile(err_bin,95);
        
    else
        
        p50(b) = NaN;
        p75(b) = NaN;
        p90(b) = NaN;
        p95(b) = NaN;
        
    end
    
    sigma_mid(b) = mean([edges(b) edges(b+1)]);
end

%% =====================================================
%% UNCERTAINTY CALIBRATION CURVE
%% =====================================================

mean_sigma_bin = zeros(Nbins,1);
mean_error_bin = zeros(Nbins,1);

for b = 1:Nbins
    
    idx_bin = sigma_pred >= edges(b) & sigma_pred < edges(b+1);
    
    if sum(idx_bin) > 10
        mean_sigma_bin(b) = mean(sigma_pred(idx_bin));
        mean_error_bin(b) = mean(errors(idx_bin));
    else
        mean_sigma_bin(b) = NaN;
        mean_error_bin(b) = NaN;
    end
end

%% =====================================================
%% RISK–COVERAGE CURVE
%% =====================================================

[sorted_sigma, sort_idx] = sort(sigma_pred,'ascend');
sorted_sq_error = errors_sq(sort_idx);
sorted_abs_error = errors(sort_idx);

N = length(sorted_sigma);

coverage_curve = (1:N)'/N;
risk_rmse_curve = sqrt(cumsum(sorted_sq_error)./(1:N)');
risk_mae_curve  = cumsum(sorted_abs_error)./(1:N)';

normalized_risk = risk_rmse_curve / baseline_rmse;

%% =====================================================
%% PLOTS
%% =====================================================

figure
plot(coverage_list*100, rmse_list,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('RMSE (sec)')
title('Coverage vs RMSE')
grid on

figure
plot(coverage_list*100, mae_list,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('MAE (sec)')
title('Coverage vs MAE')
grid on

figure
plot(coverage_curve*100, risk_rmse_curve,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('Risk (RMSE)')
title('Risk–Coverage Curve (RMSE)')
grid on

figure
plot(coverage_curve*100, normalized_risk,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('Normalized Risk')
title('Normalized Risk–Coverage Curve')
grid on

figure
plot(sigma_mid,p50,'LineWidth',2); hold on
plot(sigma_mid,p90,'LineWidth',2)
xlabel('Uncertainty (\sigma)')
ylabel('Absolute Error Percentile')
legend('Median','90th')
title('Percentile Error vs Uncertainty')
grid on

figure
plot(mean_sigma_bin, mean_error_bin,'o-','LineWidth',2); hold on
max_val = max([mean_sigma_bin; mean_error_bin]);
plot([0 max_val],[0 max_val],'k--','LineWidth',2)
xlabel('Predicted Uncertainty')
ylabel('Observed MAE')
title('Calibration Curve')
grid on

fprintf('\nReliability & Risk Analysis Completed Successfully.\n');clc; clear; close all;

fprintf('\n====================================================\n');
fprintf('   RELIABILITY & RISK ANALYSIS PIPELINE\n');
fprintf('====================================================\n');

%% =====================================================
%% LOAD MODEL
%% =====================================================

model_file = 'CNN_LSTM_REAL_MC_UNCERTAINTY_MODEL.mat';

if ~isfile(model_file)
    error('Uncertainty model not found. Run training first.');
end

load(model_file);

netCNN      = UNC_MODEL.CNN;
netLSTM     = UNC_MODEL.LSTM;
seq_len     = UNC_MODEL.seq_len;
idx_seq     = UNC_MODEL.idx_seq;
Ntrain_seq  = UNC_MODEL.Ntrain_seq;

fprintf('\nModel loaded successfully.\n');

%% =====================================================
%% LOAD DATA & REBUILD TEST SET (CORRECT SPLIT)
%% =====================================================

load FINAL_MULTI_MUSCLE_RET_CWT.mat

X = FINAL_DATA.CWT;
Y = FINAL_DATA.RET(:);
N = size(X,4);

% Recompute CNN features
features = zeros(N,64);
for i = 1:N
    features(i,:) = activations(netCNN,X(:,:,:,i),'relu_feature','OutputAs','rows');
end

% Rebuild sequences
Xseq = {};
Yseq = [];
count = 1;

for i = 1:(N-seq_len)
    Xseq{count} = features(i:i+seq_len-1,:)';
    Yseq(count,1) = Y(i+seq_len-1);
    count = count + 1;
end

% Correct test set
XTest = Xseq(idx_seq(Ntrain_seq+1:end));
YTest = Yseq(idx_seq(Ntrain_seq+1:end));

fprintf('Test samples reconstructed: %d\n',length(YTest));

%% =====================================================
%% RECOMPUTE MC DROPOUT (SAFE VERSION)
%% =====================================================

lgraph = layerGraph(netLSTM.Layers);
lgraph = removeLayers(lgraph,lgraph.Layers(end).Name);
dlnet = dlnetwork(lgraph);

Nmc = 30;
numTest = numel(XTest);

preds = zeros(numTest,Nmc);

for k = 1:Nmc
    for i = 1:numTest
        dlX = dlarray(single(XTest{i}),'CT');
        y = forward(dlnet,dlX);
        preds(i,k) = extractdata(y);
    end
end

mu_pred = mean(preds,2);

p90 = prctile(preds,90,2);
p10 = prctile(preds,10,2);
interval_width = p90 - p10;

sigma_pred = interval_width ./ (abs(mu_pred)+1e-6);
alpha = mean(abs(mu_pred - YTest)) / mean(sigma_pred);
sigma_pred = alpha * sigma_pred;

fprintf('MC Dropout recomputed successfully.\n');

%% =====================================================
%% BASELINE PERFORMANCE
%% =====================================================

errors = abs(mu_pred - YTest);
errors_sq = (mu_pred - YTest).^2;

baseline_rmse = sqrt(mean(errors_sq));
baseline_mae  = mean(errors);

fprintf('\nBaseline RMSE: %.4f sec\n',baseline_rmse);
fprintf('Baseline MAE : %.4f sec\n',baseline_mae);

%% =====================================================
%% RELIABILITY THRESHOLD ANALYSIS
%% =====================================================

T_range = linspace(min(sigma_pred), max(sigma_pred), 40);

coverage_list = zeros(size(T_range));
rmse_list = zeros(size(T_range));
mae_list = zeros(size(T_range));

Ntotal = length(YTest);

for i = 1:length(T_range)

    idx_accept = sigma_pred <= T_range(i);

    coverage_list(i) = sum(idx_accept) / Ntotal;

    if sum(idx_accept) > 10
        rmse_list(i) = sqrt(mean((mu_pred(idx_accept) - YTest(idx_accept)).^2));
        mae_list(i)  = mean(abs(mu_pred(idx_accept) - YTest(idx_accept)));
    else
        rmse_list(i) = NaN;
        mae_list(i)  = NaN;
    end
end

%% Optimal operating point (≥70% coverage)

min_coverage = 0.70;
valid_idx = coverage_list >= min_coverage;

rmse_valid = rmse_list;
rmse_valid(~valid_idx) = Inf;

[best_rmse, best_idx] = min(rmse_valid);

best_threshold = T_range(best_idx);
best_coverage  = coverage_list(best_idx);

fprintf('\n===== RELIABILITY OPERATING POINT =====\n');
fprintf('Threshold : %.4f\n', best_threshold);
fprintf('Coverage  : %.2f %%\n', best_coverage*100);
fprintf('RMSE      : %.4f sec\n', best_rmse);
fprintf('Improvement over baseline: %.2f %%\n',...
    100*(baseline_rmse-best_rmse)/baseline_rmse);

%% =====================================================
%% PERCENTILE ERROR RELIABILITY ANALYSIS
%% =====================================================

Nbins = 8;
edges = linspace(min(sigma_pred), max(sigma_pred), Nbins+1);

p50 = zeros(Nbins,1);
p75 = zeros(Nbins,1);
p90 = zeros(Nbins,1);
p95 = zeros(Nbins,1);
sigma_mid = zeros(Nbins,1);

for b = 1:Nbins
    
    idx_bin = sigma_pred >= edges(b) & sigma_pred < edges(b+1);
    
    if sum(idx_bin) > 10
        
        err_bin = errors(idx_bin);
        
        p50(b) = prctile(err_bin,50);
        p75(b) = prctile(err_bin,75);
        p90(b) = prctile(err_bin,90);
        p95(b) = prctile(err_bin,95);
        
    else
        
        p50(b) = NaN;
        p75(b) = NaN;
        p90(b) = NaN;
        p95(b) = NaN;
        
    end
    
    sigma_mid(b) = mean([edges(b) edges(b+1)]);
end

%% =====================================================
%% UNCERTAINTY CALIBRATION CURVE
%% =====================================================

mean_sigma_bin = zeros(Nbins,1);
mean_error_bin = zeros(Nbins,1);

for b = 1:Nbins
    
    idx_bin = sigma_pred >= edges(b) & sigma_pred < edges(b+1);
    
    if sum(idx_bin) > 10
        mean_sigma_bin(b) = mean(sigma_pred(idx_bin));
        mean_error_bin(b) = mean(errors(idx_bin));
    else
        mean_sigma_bin(b) = NaN;
        mean_error_bin(b) = NaN;
    end
end

%% =====================================================
%% RISK–COVERAGE CURVE
%% =====================================================

[sorted_sigma, sort_idx] = sort(sigma_pred,'ascend');
sorted_sq_error = errors_sq(sort_idx);
sorted_abs_error = errors(sort_idx);

N = length(sorted_sigma);

coverage_curve = (1:N)'/N;
risk_rmse_curve = sqrt(cumsum(sorted_sq_error)./(1:N)');
risk_mae_curve  = cumsum(sorted_abs_error)./(1:N)';

normalized_risk = risk_rmse_curve / baseline_rmse;

%% =====================================================
%% PLOTS
%% =====================================================

figure
plot(coverage_list*100, rmse_list,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('RMSE (sec)')
title('Coverage vs RMSE')
grid on

figure
plot(coverage_list*100, mae_list,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('MAE (sec)')
title('Coverage vs MAE')
grid on

figure
plot(coverage_curve*100, risk_rmse_curve,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('Risk (RMSE)')
title('Risk–Coverage Curve (RMSE)')
grid on

figure
plot(coverage_curve*100, normalized_risk,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('Normalized Risk')
title('Normalized Risk–Coverage Curve')
grid on

figure
plot(sigma_mid,p50,'LineWidth',2); hold on
plot(sigma_mid,p90,'LineWidth',2)
xlabel('Uncertainty (\sigma)')
ylabel('Absolute Error Percentile')
legend('Median','90th')
title('Percentile Error vs Uncertainty')
grid on

figure
plot(mean_sigma_bin, mean_error_bin,'o-','LineWidth',2); hold on
max_val = max([mean_sigma_bin; mean_error_bin]);
plot([0 max_val],[0 max_val],'k--','LineWidth',2)
xlabel('Predicted Uncertainty')
ylabel('Observed MAE')
title('Calibration Curve')
grid on

fprintf('\nReliability & Risk Analysis Completed Successfully.\n');