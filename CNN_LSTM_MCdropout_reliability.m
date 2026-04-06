clc; clear; close all;

fprintf('\n====================================================\n');
fprintf('   RELIABILITY & RISK ANALYSIS PIPELINE\n');
fprintf('====================================================\n');

%% =====================================================
%% LOAD SAVED MODEL
%% =====================================================

model_file = 'CNN_LSTM_REAL_MC_UNCERTAINTY_MODEL.mat';

if ~isfile(model_file)
    error('Uncertainty model not found. Run training first.');
end

load(model_file);

mu_pred    = UNC_MODEL.mu_pred;
sigma_pred = UNC_MODEL.sigma_pred;
seq_len    = UNC_MODEL.seq_len;

fprintf('\nLoaded Uncertainty Model Successfully.\n');

%% =====================================================
%% RECONSTRUCT YTest CONSISTENTLY
%% =====================================================

load FINAL_MULTI_MUSCLE_RET_CWT.mat
Y = FINAL_DATA.RET(:);

Yseq = Y(seq_len+1:end);

rng(1)
idx = randperm(length(Yseq));
Ntrain = round(0.8*length(Yseq));
test_idx = idx(Ntrain+1:end);
YTest = Yseq(test_idx);

fprintf('Total Test Samples : %d\n',length(YTest));
fprintf('----------------------------------------------------\n');

errors = abs(mu_pred - YTest);
sigma_vals = sigma_pred;

baseline_rmse = sqrt(mean((mu_pred - YTest).^2));
baseline_mae  = mean(errors);

fprintf('\nBaseline Performance:\n');
fprintf('RMSE = %.4f sec\n',baseline_rmse);
fprintf('MAE  = %.4f sec\n',baseline_mae);
fprintf('----------------------------------------------------\n');

%% =====================================================
%% RELIABILITY THRESHOLD ANALYSIS
%% =====================================================

fprintf('\nRunning Reliability Threshold Analysis...\n');

T_range = linspace(min(sigma_vals), max(sigma_vals), 40);

coverage_list = zeros(size(T_range));
rmse_list = zeros(size(T_range));
mae_list = zeros(size(T_range));

Ntotal = length(YTest);

for i = 1:length(T_range)

    T = T_range(i);
    idx_accept = sigma_vals <= T;

    coverage_list(i) = sum(idx_accept) / Ntotal;

    if sum(idx_accept) > 10
        rmse_list(i) = sqrt(mean((mu_pred(idx_accept) - YTest(idx_accept)).^2));
        mae_list(i)  = mean(abs(mu_pred(idx_accept) - YTest(idx_accept)));
    else
        rmse_list(i) = NaN;
        mae_list(i)  = NaN;
    end
end

min_coverage = 0.70;
valid_idx = coverage_list >= min_coverage;

rmse_valid = rmse_list;
rmse_valid(~valid_idx) = Inf;

[best_rmse, best_idx] = min(rmse_valid);

best_threshold = T_range(best_idx);
best_coverage  = coverage_list(best_idx);

fprintf('\n===== RELIABILITY OPERATING POINT =====\n');
fprintf('Uncertainty Threshold : %.4f\n', best_threshold);
fprintf('Coverage              : %.2f %%\n', best_coverage*100);
fprintf('RMSE (Accepted Only)  : %.4f sec\n', best_rmse);
fprintf('RMSE Improvement      : %.2f %%\n', ...
    100*(baseline_rmse - best_rmse)/baseline_rmse);
fprintf('====================================================\n');

%% =====================================================
%% UNCERTAINTY–ERROR CORRELATION
%% =====================================================

unc_corr = corr(sigma_vals, errors);
fprintf('\nUncertainty–Error Correlation = %.4f\n',unc_corr);

%% =====================================================
%% PLOTS
%% =====================================================

fprintf('\nGenerating Reliability Figures...\n');

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

idx_accept = sigma_vals <= best_threshold;

figure
scatter(YTest(idx_accept), mu_pred(idx_accept),20,'g','filled')
hold on
scatter(YTest(~idx_accept), mu_pred(~idx_accept),20,'r')
plot([min(YTest) max(YTest)],[min(YTest) max(YTest)],'k','LineWidth',2)
legend('Accepted','Rejected','Ideal','Location','best')
xlabel('True RET')
ylabel('Predicted RET')
title('Reliability-Based Filtering')
grid on

%% =====================================================
%% PERCENTILE ERROR RELIABILITY
%% =====================================================

fprintf('\nRunning Percentile Error Analysis...\n');

Nbins = 8;
edges = linspace(min(sigma_vals), max(sigma_vals), Nbins+1);

p50 = zeros(Nbins,1);
p90 = zeros(Nbins,1);
sigma_mid = zeros(Nbins,1);

for b = 1:Nbins
    idx_bin = sigma_vals >= edges(b) & sigma_vals < edges(b+1);

    if sum(idx_bin) > 10
        err_bin = errors(idx_bin);
        p50(b) = prctile(err_bin,50);
        p90(b) = prctile(err_bin,90);
    else
        p50(b) = NaN;
        p90(b) = NaN;
    end

    sigma_mid(b) = mean([edges(b) edges(b+1)]);
end

figure
plot(sigma_mid,p50,'LineWidth',2); hold on
plot(sigma_mid,p90,'LineWidth',2);
xlabel('Uncertainty (sigma)')
ylabel('Absolute Error Percentile')
legend('Median','90th Percentile')
title('Error Percentiles vs Uncertainty')
grid on

%% =====================================================
%% CALIBRATION CURVE
%% =====================================================

fprintf('\nRunning Calibration Analysis...\n');

mean_sigma_bin = zeros(Nbins,1);
mean_error_bin = zeros(Nbins,1);

for b = 1:Nbins
    idx_bin = sigma_vals >= edges(b) & sigma_vals < edges(b+1);

    if sum(idx_bin) > 10
        mean_sigma_bin(b) = mean(sigma_vals(idx_bin));
        mean_error_bin(b) = mean(errors(idx_bin));
    else
        mean_sigma_bin(b) = NaN;
        mean_error_bin(b) = NaN;
    end
end

figure
plot(mean_sigma_bin, mean_error_bin,'o-','LineWidth',2)
hold on
max_val = max([mean_sigma_bin; mean_error_bin]);
plot([0 max_val],[0 max_val],'k--','LineWidth',2)
xlabel('Predicted Uncertainty')
ylabel('Observed MAE')
title('Uncertainty Calibration Curve')
grid on

%% =====================================================
%% RISK–COVERAGE CURVE
%% =====================================================

fprintf('\nRunning Risk–Coverage Analysis...\n');

[sorted_sigma, sort_idx] = sort(sigma_vals,'ascend');
sorted_sq_error = (mu_pred(sort_idx) - YTest(sort_idx)).^2;

N = length(sorted_sigma);
coverage_curve = zeros(N,1);
risk_rmse_curve = zeros(N,1);

for k = 10:N
    coverage_curve(k) = k/N;
    risk_rmse_curve(k) = sqrt(mean(sorted_sq_error(1:k)));
end

figure
plot(coverage_curve*100, risk_rmse_curve,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('Risk (RMSE)')
title('Risk–Coverage Curve')
grid on

fprintf('\nReliability Analysis Completed Successfully.\n');