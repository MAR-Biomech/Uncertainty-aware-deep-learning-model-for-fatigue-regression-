clc; clear; close all;

fprintf('\n====================================================\n');
fprintf(' CNN + LSTM + MC DROPOUT UNCERTAINTY PIPELINE\n');
fprintf('====================================================\n');

model_file = 'CNN_LSTM_REAL_MC_UNCERTAINTY_MODEL.mat';

%% =====================================================
%% LOAD DATA
%% =====================================================
fprintf('\nLoading dataset...\n');
load FINAL_MULTI_MUSCLE_RET_CWT.mat

X = FINAL_DATA.CWT;
Y = FINAL_DATA.RET(:);
N = size(X,4);

fprintf('Total Samples : %d\n', N);
fprintf('Input Size    : %d x %d x %d\n', size(X,1), size(X,2), size(X,3));
fprintf('RET Range     : %.2f to %.2f sec\n', min(Y), max(Y));
fprintf('----------------------------------------------------\n');

%% =====================================================
%% TRAIN OR LOAD MODEL
%% =====================================================
if isfile(model_file)

    fprintf('\nLoading saved model...\n');
    load(model_file);

    netCNN      = UNC_MODEL.CNN;
    netLSTM     = UNC_MODEL.LSTM;
    seq_len     = UNC_MODEL.seq_len;
    idx_seq     = UNC_MODEL.idx_seq;
    Ntrain_seq  = UNC_MODEL.Ntrain_seq;

    fprintf('Model loaded successfully.\n');

else

    fprintf('\nNo saved model found. Training from scratch...\n');

    %% ================= CNN =================
    layersCNN = [
        imageInputLayer([64 64 1])
        convolution2dLayer(3,16,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2)

        convolution2dLayer(3,32,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2)

        convolution2dLayer(3,64,'Padding','same')
        batchNormalizationLayer
        reluLayer
        maxPooling2dLayer(2)

        flattenLayer
        fullyConnectedLayer(64)
        reluLayer('Name','relu_feature')
        fullyConnectedLayer(1)
        regressionLayer];

    rng(1)
    idx = randperm(N);
    Ntrain = round(0.8*N);

    XTrainCNN = X(:,:,:,idx(1:Ntrain));
    YTrainCNN = Y(idx(1:Ntrain));

    optionsCNN = trainingOptions('adam',...
        'MaxEpochs',15,...
        'MiniBatchSize',32,...
        'Verbose',false);

    netCNN = trainNetwork(XTrainCNN,YTrainCNN,layersCNN,optionsCNN);
    fprintf('CNN training complete.\n');

    %% ================= FEATURE EXTRACTION =================
    features = zeros(N,64);
    for i = 1:N
        features(i,:) = activations(netCNN,X(:,:,:,i),'relu_feature','OutputAs','rows');
    end

    %% ================= BUILD SEQUENCES =================
    seq_len = 8;
    Xseq = {};
    Yseq = [];
    count = 1;

    for i = 1:(N-seq_len)
        Xseq{count} = features(i:i+seq_len-1,:)';
        Yseq(count,1) = Y(i+seq_len-1);
        count = count + 1;
    end

    %% ================= SPLIT =================
    rng(1)
    Nseq = length(Yseq);
    idx_seq = randperm(Nseq);
    Ntrain_seq = round(0.8*Nseq);

    XTrain = Xseq(idx_seq(1:Ntrain_seq));
    YTrain = Yseq(idx_seq(1:Ntrain_seq));

    %% ================= LSTM + DROPOUT =================
    numFeatures = size(XTrain{1},1);

    layersLSTM = [
        sequenceInputLayer(numFeatures)
        lstmLayer(64,'OutputMode','last')
        dropoutLayer(0.3)
        fullyConnectedLayer(32)
        reluLayer
        dropoutLayer(0.3)
        fullyConnectedLayer(1)
        regressionLayer];

    optionsLSTM = trainingOptions('adam',...
        'MaxEpochs',40,...
        'MiniBatchSize',16,...
        'Verbose',false);

    netLSTM = trainNetwork(XTrain,YTrain,layersLSTM,optionsLSTM);
    fprintf('LSTM training complete.\n');

    %% SAVE MODEL
    UNC_MODEL.CNN = netCNN;
    UNC_MODEL.LSTM = netLSTM;
    UNC_MODEL.seq_len = seq_len;
    UNC_MODEL.idx_seq = idx_seq;
    UNC_MODEL.Ntrain_seq = Ntrain_seq;

    save(model_file,'UNC_MODEL','-v7.3');
    fprintf('Model saved successfully.\n');
end

%% =====================================================
%% REBUILD TEST DATA
%% =====================================================

features = zeros(N,64);
for i = 1:N
    features(i,:) = activations(netCNN,X(:,:,:,i),'relu_feature','OutputAs','rows');
end

Xseq = {};
Yseq = [];
count = 1;

for i = 1:(N-seq_len)
    Xseq{count} = features(i:i+seq_len-1,:)';
    Yseq(count,1) = Y(i+seq_len-1);
    count = count + 1;
end

XTest = Xseq(idx_seq(Ntrain_seq+1:end));
YTest = Yseq(idx_seq(Ntrain_seq+1:end));

fprintf('\nTest Samples: %d\n', length(YTest));

%% =====================================================
%% MC DROPOUT INFERENCE
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

%% =====================================================
%% UNCERTAINTY ESTIMATION
%% =====================================================

p90 = prctile(preds,90,2);
p10 = prctile(preds,10,2);
interval_width = p90 - p10;

sigma_pred = interval_width ./ (abs(mu_pred)+1e-6);
alpha = mean(abs(mu_pred - YTest)) / mean(sigma_pred);
sigma_pred = alpha * sigma_pred;

%% =====================================================
%% PERFORMANCE METRICS
%% =====================================================

errors = mu_pred - YTest;

RMSE = sqrt(mean(errors.^2));
MAE  = mean(abs(errors));
R2   = 1 - sum(errors.^2)/sum((YTest-mean(YTest)).^2);
r    = corr(YTest,mu_pred);

NRMSE = RMSE/(max(YTest)-min(YTest));
MAPE = mean(abs(errors./YTest))*100;

fprintf('\n================ PERFORMANCE =================\n');
fprintf('RMSE                : %.4f sec\n',RMSE);
fprintf('MAE                 : %.4f sec\n',MAE);
fprintf('NRMSE               : %.4f\n',NRMSE);
fprintf('MAPE                : %.2f %%\n',MAPE);
fprintf('R^2                 : %.4f\n',R2);
fprintf('Correlation (r)     : %.4f\n',r);
fprintf('Residual Mean       : %.4f\n',mean(errors));
fprintf('Residual Std        : %.4f\n',std(errors));
fprintf('Residual Max        : %.4f\n',max(errors));
fprintf('Residual Min        : %.4f\n',min(errors));
fprintf('================================================\n');

%% =====================================================
%% UNCERTAINTY METRICS
%% =====================================================

unc_error_corr = corr(sigma_pred,abs(errors));

fprintf('\n============== UNCERTAINTY =================\n');
fprintf('Mean Uncertainty        : %.4f\n',mean(sigma_pred));
fprintf('Uncertainty-Error Corr  : %.4f\n',unc_error_corr);
fprintf('=============================================\n');

%% =====================================================
%% VISUALIZATION
%% =====================================================

fprintf('\nGenerating evaluation plots...\n');

figure
scatter(YTest,mu_pred,25,'filled')
hold on
plot([min(YTest) max(YTest)],[min(YTest) max(YTest)],'r','LineWidth',2)
xlabel('True RET')
ylabel('Predicted RET')
title(sprintf('Prediction (R^2 = %.3f)',R2))
grid on

figure
scatter(YTest,errors,25,'filled')
yline(0,'r','LineWidth',2)
xlabel('True RET')
ylabel('Residual')
title('Residual vs True')
grid on

figure
scatter(mu_pred,errors,25,'filled')
yline(0,'r','LineWidth',2)
xlabel('Predicted RET')
ylabel('Residual')
title('Residual vs Predicted')
grid on

figure
histogram(errors,30)
title('Residual Distribution')
xlabel('Error')
ylabel('Frequency')
grid on

figure
scatter(sigma_pred,abs(errors),25,'filled')
xlabel('Uncertainty')
ylabel('Absolute Error')
title('Uncertainty vs Error')
grid on

%% CDF ERROR
abs_errors = abs(errors);
sorted_err = sort(abs_errors);
cdf_vals = (1:length(sorted_err))/length(sorted_err);

figure
plot(sorted_err,cdf_vals,'LineWidth',2)
xlabel('Absolute Error')
ylabel('CDF')
title('Cumulative Error Distribution')
grid on

%% BLAND ALTMAN
mean_vals = (YTest + mu_pred)/2;
diff_vals = errors;

mean_diff = mean(diff_vals);
std_diff  = std(diff_vals);

upper = mean_diff + 1.96*std_diff;
lower = mean_diff - 1.96*std_diff;

figure
scatter(mean_vals,diff_vals,25,'filled')
hold on
yline(mean_diff,'k','LineWidth',2)
yline(upper,'r--','LineWidth',2)
yline(lower,'r--','LineWidth',2)
xlabel('Mean of True & Predicted')
ylabel('Difference')
title('Bland-Altman Analysis')
grid on

%% =====================================================
%% SAVE RESULTS TO EXCEL
%% =====================================================

fprintf('\nSaving results to Excel...\n');

results_table = table(YTest,mu_pred,errors,abs(errors),sigma_pred,...
    'VariableNames',{'True_RET','Pred_RET','Error','Abs_Error','Uncertainty'});

metrics_table = table(RMSE,MAE,NRMSE,MAPE,R2,r,...
    mean(errors),std(errors),max(errors),min(errors),...
    'VariableNames',{'RMSE','MAE','NRMSE','MAPE','R2','Corr',...
    'MeanResidual','StdResidual','MaxResidual','MinResidual'});

excel_file = 'CNN_LSTM_MC_UNCERTAINTY_RESULTS.xlsx';

writetable(results_table,excel_file,'Sheet','Predictions');
writetable(metrics_table,excel_file,'Sheet','Metrics');

fprintf('Excel file saved: %s\n',excel_file);

fprintf('\nPipeline Completed Successfully.\n');
%%
%% =====================================================
%% SAVE RESULTS TO EXCEL
%% =====================================================

fprintf('\nSaving results to Excel...\n');

excel_file = 'CNN_LSTM_MC_Uncertainty_Results.xlsx';

%% -----------------------------------------------------
%% Prediction Results
%% -----------------------------------------------------

results_table = table;

results_table.True_RET       = YTest;
results_table.Predicted_RET  = mu_pred;
results_table.Error          = errors;
results_table.Abs_Error      = abs(errors);
results_table.Uncertainty    = sigma_pred;

writetable(results_table,excel_file,'Sheet','Predictions');

%% -----------------------------------------------------
%% Performance Metrics
%% -----------------------------------------------------

metrics_table = table;

metrics_table.RMSE = RMSE;
metrics_table.MAE  = MAE;
metrics_table.R2   = R2;
metrics_table.Correlation = r;
metrics_table.MeanResidual = mean(errors);
metrics_table.StdResidual  = std(errors);
metrics_table.MaxResidual  = max(errors);
metrics_table.MinResidual  = min(errors);

writetable(metrics_table,excel_file,'Sheet','Metrics');

%% -----------------------------------------------------
%% MC Dropout Statistics
%% -----------------------------------------------------

mc_table = table;

mc_table.Mean_Uncertainty = mean(sigma_pred);
mc_table.Std_Uncertainty  = std(sigma_pred);
mc_table.Min_Uncertainty  = min(sigma_pred);
mc_table.Max_Uncertainty  = max(sigma_pred);

writetable(mc_table,excel_file,'Sheet','Uncertainty_Stats');

%% -----------------------------------------------------
%% Optional: Save Raw MC Predictions
%% -----------------------------------------------------

preds_table = array2table(preds);
writetable(preds_table,excel_file,'Sheet','MC_Predictions');

fprintf('Excel file saved: %s\n',excel_file);
fprintf('================================================\n');
%%%% =====================================================

%% =====================================================
%% CNN-LSTM MC DROPOUT VISUALIZATION
%% =====================================================

fprintf('\nGenerating CNN-LSTM MC Dropout evaluation plots...\n');

set(groot,'defaultAxesFontSize',12);
set(groot,'defaultAxesFontName','Times New Roman');

%% =====================================================
%% FIGURE 1 : PREDICTION PERFORMANCE
%% =====================================================

figure('Color','w','Position',[200 200 900 420]);
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

%% (a) TRUE vs PREDICTED
nexttile

scatter(YTest,mu_pred,35,'filled','MarkerFaceAlpha',0.7)
hold on

plot([min(YTest) max(YTest)],...
     [min(YTest) max(YTest)],...
     'r','LineWidth',2)

xlabel('True RET (sec)')
ylabel('Predicted RET (sec)')
title('(a) True vs Predicted')

grid on
axis square

text(min(YTest)+2,max(YTest)-5,...
    sprintf('R^2 = %.3f\nr = %.3f',R2,r),...
    'FontWeight','bold')

%% (b) REGRESSION FIT
nexttile

p = polyfit(YTest,mu_pred,1);
yfit = polyval(p,YTest);

scatter(YTest,mu_pred,35,'filled','MarkerFaceAlpha',0.7)
hold on

plot(YTest,yfit,'k','LineWidth',2)

xlabel('True RET (sec)')
ylabel('Predicted RET (sec)')
title('(b) Regression Fit')

grid on
axis square

text(min(YTest)+2,max(YTest)-5,...
    sprintf('Slope = %.3f',p(1)),...
    'FontWeight','bold')

sgtitle('CNN-LSTM MC Dropout Prediction Performance')


%% =====================================================
%% FIGURE 2 : ERROR & UNCERTAINTY ANALYSIS
%% =====================================================

figure('Color','w','Position',[200 200 1000 420]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

%% (a) RESIDUAL HISTOGRAM
nexttile

histogram(errors,30,...
    'FaceColor',[0.2 0.4 0.8],...
    'EdgeColor','none')

xlabel('Prediction Error (sec)')
ylabel('Frequency')
title('(a) Residual Distribution')

grid on


%% (b) BLAND–ALTMAN
nexttile

mean_vals = (YTest + mu_pred)/2;
diff_vals = errors;

mean_diff = mean(diff_vals);
std_diff  = std(diff_vals);

upper = mean_diff + 1.96*std_diff;
lower = mean_diff - 1.96*std_diff;

scatter(mean_vals,diff_vals,35,'filled','MarkerFaceAlpha',0.7)
hold on

yline(mean_diff,'k','LineWidth',2)
yline(upper,'r--','LineWidth',2)
yline(lower,'r--','LineWidth',2)

xlabel('Mean RET (sec)')
ylabel('Prediction Difference (sec)')
title('(b) Bland–Altman Plot')

grid on


%% (c) UNCERTAINTY vs ERROR
nexttile

scatter(sigma_pred,abs(errors),35,'filled','MarkerFaceAlpha',0.7)

xlabel('Predictive Uncertainty')
ylabel('Absolute Error')

title('(c) Uncertainty vs Error')

grid on


%% (d) CUMULATIVE ERROR DISTRIBUTION

nexttile

abs_errors = abs(errors);
sorted_err = sort(abs_errors);
cdf_vals = (1:length(sorted_err))/length(sorted_err);

plot(sorted_err,cdf_vals,'LineWidth',2)

xlabel('Absolute Error (sec)')
ylabel('Cumulative Probability')

title('(d) Cumulative Absolute Error')

grid on

sgtitle('CNN-LSTM MC Dropout Error and Uncertainty Analysis')

fprintf('CNN-LSTM MC Dropout plots generated successfully.\n');
%% IMPROVED RELIABILITY ANALYSIS (RISK–COVERAGE CURVE)
%% =====================================================

fprintf('\n====================================================\n');
fprintf('Running Improved Reliability Analysis\n');
fprintf('====================================================\n');

Ntotal = length(YTest);

%% -----------------------------------------------------
%% SORT SAMPLES BY UNCERTAINTY
%% -----------------------------------------------------

[sorted_sigma, idx_sorted] = sort(sigma_pred);

sorted_errors = abs(errors(idx_sorted));
sorted_sqerr  = errors(idx_sorted).^2;

%% -----------------------------------------------------
%% COVERAGE STEPS
%% -----------------------------------------------------

coverage_vec = zeros(Ntotal,1);
rmse_vec     = zeros(Ntotal,1);
mae_vec      = zeros(Ntotal,1);

for k = 5:Ntotal
    
    coverage_vec(k) = k / Ntotal;
    
    rmse_vec(k) = sqrt(mean(sorted_sqerr(1:k)));
    
    mae_vec(k) = mean(sorted_errors(1:k));
    
end

%% -----------------------------------------------------
%% RELIABILITY OPERATING POINT (90% COVERAGE)
%% -----------------------------------------------------

target_cov = 0.90;

[~,idx_op] = min(abs(coverage_vec-target_cov));

coverage_opt = coverage_vec(idx_op);
rmse_opt     = rmse_vec(idx_op);
mae_opt      = mae_vec(idx_op);

fprintf('\n===== RELIABILITY OPERATING POINT =====\n');
fprintf('Coverage              : %.2f %%\n',coverage_opt*100);
fprintf('Accepted Samples      : %d / %d\n',idx_op,Ntotal);
fprintf('RMSE (Accepted only)  : %.4f sec\n',rmse_opt);
fprintf('MAE  (Accepted only)  : %.4f sec\n',mae_opt);
fprintf('=======================================\n');

%% -----------------------------------------------------
%% RELIABILITY CURVE PLOTS
%% -----------------------------------------------------

figure('Name','Risk Coverage RMSE','Color','w')
plot(coverage_vec*100,rmse_vec,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('RMSE (sec)')
title('Risk–Coverage Curve (RMSE)')
grid on

figure('Name','Risk Coverage MAE','Color','w')
plot(coverage_vec*100,mae_vec,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('MAE (sec)')
title('Risk–Coverage Curve (MAE)')
grid on

%% -----------------------------------------------------
%% SAVE RELIABILITY RESULTS
%% -----------------------------------------------------

reliability_table = table(coverage_vec*100,rmse_vec,mae_vec,...
    'VariableNames',{'Coverage_percent','RMSE','MAE'});

excel_file = 'Improved_Reliability_RET.xlsx';

writetable(reliability_table,excel_file,'Sheet','RiskCoverage');

fprintf('\nReliability results saved to: %s\n',excel_file);

fprintf('\nImproved Reliability Analysis Completed.\n');
%% =====================================================
%% RELIABILITY ANALYSIS
%% =====================================================

fprintf('\nRunning Reliability Analysis...\n');

thresholds = linspace(min(sigma_pred),max(sigma_pred),50);

coverage_vec = zeros(length(thresholds),1);
rmse_vec = zeros(length(thresholds),1);
mae_vec = zeros(length(thresholds),1);

Ntotal = length(YTest);

for t = 1:length(thresholds)

    T = thresholds(t);

    accepted = sigma_pred < T;

    coverage_vec(t) = sum(accepted)/Ntotal;

    if sum(accepted) > 5

        err = errors(accepted);

        rmse_vec(t) = sqrt(mean(err.^2));
        mae_vec(t)  = mean(abs(err));

    else

        rmse_vec(t) = NaN;
        mae_vec(t)  = NaN;

    end

end

%% =====================================================
%% SELECT RELIABILITY OPERATING POINT
%% =====================================================

target_cov = 0.90;

[~,idx] = min(abs(coverage_vec-target_cov));

T_opt = thresholds(idx);

accepted = sigma_pred < T_opt;

coverage_opt = sum(accepted)/Ntotal;

rmse_opt = sqrt(mean(errors(accepted).^2));
mae_opt  = mean(abs(errors(accepted)));

fprintf('\n===== RELIABILITY OPERATING POINT =====\n');
fprintf('Uncertainty Threshold : %.4f\n',T_opt);
fprintf('Coverage              : %.2f %%\n',coverage_opt*100);
fprintf('RMSE (Accepted only)  : %.4f sec\n',rmse_opt);
fprintf('MAE  (Accepted only)  : %.4f sec\n',mae_opt);
fprintf('Accepted Samples      : %d / %d\n',sum(accepted),Ntotal);
fprintf('=======================================\n');

%% =====================================================
%% RELIABILITY CURVE ANALYSIS
%% =====================================================

fprintf('\n====================================================\n');
fprintf('Running Reliability Curve Analysis\n');
fprintf('====================================================\n');

Ntotal = length(YTest);

%% Sweep uncertainty thresholds
num_points = 50;
thresholds = linspace(min(sigma_pred), max(sigma_pred), num_points);

coverage_vec = zeros(num_points,1);
rmse_vec = zeros(num_points,1);
mae_vec = zeros(num_points,1);
accepted_samples = zeros(num_points,1);

for i = 1:num_points
    
    T = thresholds(i);
    
    % Accept predictions with uncertainty below threshold
    accepted = sigma_pred <= T;
    
    Naccepted = sum(accepted);
    
    accepted_samples(i) = Naccepted;
    
    coverage_vec(i) = Naccepted / Ntotal;
    
    if Naccepted > 5
        
        err = errors(accepted);
        
        rmse_vec(i) = sqrt(mean(err.^2));
        mae_vec  (i) = mean(abs(err));
        
    else
        
        rmse_vec(i) = NaN;
        mae_vec(i) = NaN;
        
    end
    
end

fprintf('\nReliability curve computed using %d thresholds.\n',num_points);

%% =====================================================
%% RELIABILITY OPERATING POINT (90% COVERAGE)
%% =====================================================

target_coverage = 0.90;

[~,idx_opt] = min(abs(coverage_vec - target_coverage));

T_opt = thresholds(idx_opt);

accepted_opt = sigma_pred <= T_opt;

coverage_opt = coverage_vec(idx_opt);

rmse_opt = sqrt(mean(errors(accepted_opt).^2));
mae_opt  = mean(abs(errors(accepted_opt)));

fprintf('\n===== RELIABILITY OPERATING POINT =====\n');
fprintf('Uncertainty Threshold : %.4f\n',T_opt);
fprintf('Coverage              : %.2f %%\n',coverage_opt*100);
fprintf('Accepted Samples      : %d / %d\n',sum(accepted_opt),Ntotal);
fprintf('RMSE (Accepted only)  : %.4f sec\n',rmse_opt);
fprintf('MAE  (Accepted only)  : %.4f sec\n',mae_opt);
fprintf('=======================================\n');

%% =====================================================
%% RELIABILITY CURVE PLOTS
%% =====================================================

fprintf('\nGenerating Reliability Curves...\n');

% Coverage vs RMSE
figure('Name','Reliability Curve RMSE','Color','w');
plot(coverage_vec*100, rmse_vec,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('RMSE (sec)')
title('Reliability Curve (Coverage vs RMSE)')
grid on

% Coverage vs MAE
figure('Name','Reliability Curve MAE','Color','w');
plot(coverage_vec*100, mae_vec,'LineWidth',2)
xlabel('Coverage (%)')
ylabel('MAE (sec)')
title('Reliability Curve (Coverage vs MAE)')
grid on

%% =====================================================
%% RELIABILITY TABLE
%% =====================================================

reliability_table = table( ...
    thresholds', ...
    coverage_vec*100, ...
    accepted_samples, ...
    rmse_vec, ...
    mae_vec, ...
    'VariableNames',{'Threshold','Coverage_percent','Accepted_samples','RMSE','MAE'});

fprintf('\nFirst 10 rows of Reliability Table:\n');
disp(reliability_table(1:min(10,height(reliability_table)),:));

%% =====================================================
%% SAVE RELIABILITY RESULTS TO EXCEL
%% =====================================================

excel_file = 'Reliability_Analysis_RET.xlsx';

writetable(reliability_table,excel_file,'Sheet','Reliability_Curve');

operating_point_table = table(T_opt,coverage_opt*100,rmse_opt,mae_opt,...
    'VariableNames',{'Threshold','Coverage_percent','RMSE','MAE'});

writetable(operating_point_table,excel_file,'Sheet','Operating_Point');

fprintf('\nReliability results saved to: %s\n',excel_file);

fprintf('\nReliability Analysis Completed Successfully.\n');
%% =====================================================
%% AURC (AREA UNDER RISK–COVERAGE CURVE)
%% =====================================================

fprintf('\n====================================================\n');
fprintf('Computing AURC (Area Under Risk–Coverage Curve)\n');
fprintf('====================================================\n');

% Convert coverage to fraction
coverage_frac = coverage_vec;

% Remove NaNs
valid = ~isnan(rmse_vec);

cov_valid = coverage_frac(valid);
rmse_valid = rmse_vec(valid);
mae_valid = mae_vec(valid);

% Compute area using trapezoidal rule
AURC_RMSE = trapz(cov_valid, rmse_valid);
AURC_MAE  = trapz(cov_valid, mae_valid);

fprintf('\n===== RELIABILITY SCORE =====\n');
fprintf('AURC (RMSE) : %.4f\n',AURC_RMSE);
fprintf('AURC (MAE)  : %.4f\n',AURC_MAE);
fprintf('Lower AURC indicates better reliability.\n');
fprintf('=======================================\n');