clc; clear; close all;

fprintf('\n============================================\n');
fprintf('   CNN RET FATIGUE PREDICTION PIPELINE\n');
fprintf('============================================\n');

%% ================= LOAD DATA =================
fprintf('\nLoading dataset...\n');
load FINAL_MULTI_MUSCLE_RET_CWT.mat

X = FINAL_DATA.CWT;
Y = FINAL_DATA.RET(:);

fprintf('Dataset Loaded Successfully.\n');
fprintf('Total Samples: %d\n', size(X,4));
fprintf('Input Size: %d x %d x %d\n', size(X,1), size(X,2), size(X,3));
fprintf('RET Range: %.2f to %.2f sec\n', min(Y), max(Y));

%% ================= TRAIN TEST SPLIT =================
rng(1)

N = size(X,4);
idx = randperm(N);

train_ratio = 0.8;
Ntrain = round(train_ratio*N);

train_idx = idx(1:Ntrain);
test_idx  = idx(Ntrain+1:end);

XTrain = X(:,:,:,train_idx);
YTrain = Y(train_idx);

XTest = X(:,:,:,test_idx);
YTest = Y(test_idx);

fprintf('\nTrain/Test Split Completed.\n');
fprintf('Training Samples: %d\n', length(YTrain));
fprintf('Testing Samples : %d\n', length(YTest));

%% =====================================================
%% CHECK IF MODEL EXISTS
%% =====================================================

model_file = 'CNN_RET_BASELINE_MODEL.mat';

if isfile(model_file)

    fprintf('\nExisting trained model found.\n');
    fprintf('Loading model...\n');

    load(model_file);

    net = CNN_MODEL.net;

    fprintf('Model loaded successfully!\n');
    fprintf('Stored RMSE = %.3f sec\n', CNN_MODEL.RMSE);
    fprintf('Stored MAE  = %.3f sec\n', CNN_MODEL.MAE);

else

    fprintf('\nNo trained model found.\n');
    fprintf('Starting CNN training...\n');

    %% ================= CNN ARCHITECTURE =================
    layers = [

    imageInputLayer([64 64 1])

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    flattenLayer

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.4)

    fullyConnectedLayer(1)
    regressionLayer
    ];

    fprintf('CNN Architecture Defined.\n');
    fprintf('Total Learnable Parameters will be displayed during training.\n');

    %% ================= TRAINING OPTIONS =================
    options = trainingOptions('adam', ...
        'MaxEpochs',40, ...
        'MiniBatchSize',32, ...
        'InitialLearnRate',1e-3, ...
        'Shuffle','every-epoch', ...
        'Plots','training-progress', ...
        'Verbose',true);

    fprintf('\nTraining Configuration:\n');
    fprintf('Epochs: %d\n', 40);
    fprintf('MiniBatch Size: %d\n', 32);
    fprintf('Initial Learning Rate: %.4f\n', 1e-3);
    fprintf('Optimizer: Adam\n');

    %% ================= TRAIN =================
    net = trainNetwork(XTrain,YTrain,layers,options);

    fprintf('\nTraining Completed Successfully.\n');

    %% ================= TEST =================
    fprintf('\nEvaluating on Test Set...\n');

    YPred = predict(net,XTest);
%% ================= PERFORMANCE METRICS =================

% --- Errors ---
errors = YPred - YTest;
residuals = errors;

% --- RMSE ---
RMSE = sqrt(mean(errors.^2));

% --- MAE ---
MAE = mean(abs(errors));

% --- R^2 ---
SS_res = sum((YTest - YPred).^2);
SS_tot = sum((YTest - mean(YTest)).^2);
R2 = 1 - (SS_res / SS_tot);

% --- Pearson Correlation ---
r = corr(YTest, YPred);

%% ================= PRINT RESULTS =================
fprintf('\n===== TEST PERFORMANCE =====\n');
fprintf('RMSE  = %.4f sec\n', RMSE);
fprintf('MAE   = %.4f sec\n', MAE);
fprintf('R^2   = %.4f\n', R2);
fprintf('Corr  = %.4f\n', r);
fprintf('=================================\n');

%% ================= RESIDUAL ANALYSIS =================

fprintf('\nResidual Statistics:\n');
fprintf('Mean Residual  = %.4f sec\n', mean(residuals));
fprintf('Std Residual   = %.4f sec\n', std(residuals));
fprintf('Max Residual   = %.4f sec\n', max(residuals));
fprintf('Min Residual   = %.4f sec\n', min(residuals));

%% ================= VISUALIZATION =================

figure('Name','True vs Predicted RET')
scatter(YTest, YPred, 25, 'filled')
hold on
plot([min(YTest) max(YTest)], [min(YTest) max(YTest)], 'r', 'LineWidth', 2)
xlabel('True RET (sec)')
ylabel('Predicted RET (sec)')
title('CNN RET Prediction')
grid on

figure('Name','Residual Histogram')
histogram(residuals,30)
xlabel('Prediction Error (sec)')
ylabel('Frequency')
title('Residual Distribution')
grid on

figure('Name','Residual vs True RET')
scatter(YTest, residuals, 25, 'filled')
yline(0,'r','LineWidth',2)
xlabel('True RET (sec)')
ylabel('Residual (sec)')
title('Residual vs True RET')
grid on
    RMSE = sqrt(mean((YPred - YTest).^2));
    MAE  = mean(abs(YPred - YTest));

    fprintf('\n===== TEST RESULTS =====\n');
    fprintf('RMSE = %.3f sec\n',RMSE);
    fprintf('MAE  = %.3f sec\n',MAE);

    %% ================= SAVE MODEL =================
    fprintf('\nSaving trained model...\n');

    CNN_MODEL.net = net;
    CNN_MODEL.train_idx = train_idx;
    CNN_MODEL.test_idx  = test_idx;
    CNN_MODEL.RMSE = RMSE;
    CNN_MODEL.MAE  = MAE;
    CNN_MODEL.train_ratio = train_ratio;

    save(model_file,'CNN_MODEL','-v7.3');

    fprintf('Model saved successfully!\n');

end

%% ================= FINAL EVALUATION =================

fprintf('\nRunning evaluation with loaded/trained model...\n');

YPred = predict(net,XTest);

RMSE = sqrt(mean((YPred - YTest).^2));
MAE  = mean(abs(YPred - YTest));

fprintf('\n===== FINAL TEST RESULTS =====\n');
fprintf('RMSE = %.3f sec\n',RMSE);
fprintf('MAE  = %.3f sec\n',MAE);

%% ================= SCATTER PLOT =================

figure
scatter(YTest,YPred,20,'filled')
hold on
plot([min(YTest) max(YTest)],[min(YTest) max(YTest)],'r','LineWidth',2)
xlabel('True RET (sec)')
ylabel('Predicted RET (sec)')
title('CNN RET Prediction')
grid on

fprintf('\nPipeline Finished Successfully.\n');
fprintf('============================================\n');
%% ================= FINAL EVALUATION =================

fprintf('\nRunning evaluation with trained CNN model...\n');

YPred = predict(net,XTest);

%% =====================================================
%% PERFORMANCE METRICS
%% =====================================================

errors = YPred - YTest;

RMSE = sqrt(mean(errors.^2));
MAE  = mean(abs(errors));

SS_res = sum((YTest - YPred).^2);
SS_tot = sum((YTest - mean(YTest)).^2);
R2 = 1 - SS_res/SS_tot;

r = corr(YTest,YPred);

fprintf('\n================ PERFORMANCE =================\n');
fprintf('RMSE                : %.4f sec\n', RMSE);
fprintf('MAE                 : %.4f sec\n', MAE);
fprintf('R^2                 : %.4f\n', R2);
fprintf('Correlation (r)     : %.4f\n', r);
fprintf('------------------------------------------------\n');
fprintf('Residual Mean       : %.4f\n', mean(errors));
fprintf('Residual Std        : %.4f\n', std(errors));
fprintf('Residual Max        : %.4f\n', max(errors));
fprintf('Residual Min        : %.4f\n', min(errors));
fprintf('================================================\n');

%% =====================================================
%% VISUALIZATION
%% =====================================================

fprintf('\nGenerating evaluation plots...\n');

%% TRUE vs PREDICTED
figure('Name','True vs Predicted RET','Color','w');
scatter(YTest, YPred, 25, 'filled');
hold on
plot([min(YTest) max(YTest)],...
     [min(YTest) max(YTest)],...
     'r','LineWidth',2);
xlabel('True RET (sec)');
ylabel('Predicted RET (sec)');
title(sprintf('CNN RET Prediction (R^2 = %.3f)',R2));
grid on

%% REGRESSION LINE
p = polyfit(YTest, YPred, 1);
yfit = polyval(p, YTest);

figure('Name','Regression Fit','Color','w');
scatter(YTest, YPred, 20, 'filled'); hold on
plot(YTest, yfit, 'k','LineWidth',2);
xlabel('True RET (sec)');
ylabel('Predicted RET (sec)');
title('Linear Fit Between True and Predicted');
grid on

%% RESIDUAL HISTOGRAM
figure('Name','Residual Distribution','Color','w');
histogram(errors,30);
xlabel('Prediction Error (sec)');
ylabel('Frequency');
title('Residual Histogram');
grid on

%% RESIDUAL vs TRUE
figure('Name','Residual vs True RET','Color','w');
scatter(YTest, errors, 25, 'filled');
yline(0,'r','LineWidth',2);
xlabel('True RET (sec)');
ylabel('Residual (sec)');
title('Residual vs True RET');
grid on

%% RESIDUAL vs PREDICTED
figure('Name','Residual vs Predicted RET','Color','w');
scatter(YPred, errors, 25, 'filled');
yline(0,'r','LineWidth',2);
xlabel('Predicted RET (sec)');
ylabel('Residual (sec)');
title('Residual vs Predicted RET');
grid on

%% CUMULATIVE ERROR DISTRIBUTION
abs_errors = abs(errors);
sorted_err = sort(abs_errors);
cdf_vals = (1:length(sorted_err))/length(sorted_err);

figure('Name','Cumulative Error Distribution','Color','w');
plot(sorted_err, cdf_vals,'LineWidth',2);
xlabel('Absolute Error (sec)');
ylabel('Cumulative Probability');
title('Cumulative Absolute Error');
grid on

%% BLAND ALTMAN
mean_vals = (YTest + YPred)/2;
diff_vals = errors;

mean_diff = mean(diff_vals);
std_diff  = std(diff_vals);

upper = mean_diff + 1.96*std_diff;
lower = mean_diff - 1.96*std_diff;

figure('Name','Bland-Altman Plot','Color','w');
scatter(mean_vals, diff_vals,25,'filled'); hold on
yline(mean_diff,'k','LineWidth',2);
yline(upper,'r--','LineWidth',2);
yline(lower,'r--','LineWidth',2);
xlabel('Mean of True and Predicted RET');
ylabel('Difference (sec)');
title('Bland-Altman Analysis');
grid on

fprintf('All plots generated successfully.\n');
%% =====================================================
%% SAVE RESULTS TO EXCEL
%% =====================================================

fprintf('\nSaving results to Excel...\n');

results_table = table;

results_table.True_RET = YTest;
results_table.Predicted_RET = YPred;
results_table.Error = errors;
results_table.Abs_Error = abs(errors);

excel_file = 'CNN_RET_RESULTS.xlsx';

writetable(results_table, excel_file, 'Sheet', 'Predictions');

metrics_table = table(RMSE,MAE,R2,r,...
    mean(errors),std(errors),max(errors),min(errors),...
    'VariableNames',{'RMSE','MAE','R2','Correlation','MeanResidual','StdResidual','MaxResidual','MinResidual'});

writetable(metrics_table, excel_file, 'Sheet', 'Metrics');

fprintf('Excel file saved: %s\n', excel_file);
%%
%% =====================================================
%% IEEE-STYLE VISUALIZATION
%% =====================================================

fprintf('\nGenerating IEEE-quality evaluation plots...\n');

set(groot,'defaultAxesFontSize',12);
set(groot,'defaultAxesFontName','Times New Roman');

%% =====================================================
%% FIGURE 1 : PREDICTION PERFORMANCE
%% =====================================================

figure('Color','w','Position',[200 200 900 400]);

tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

%% ---------- (a) TRUE vs PREDICTED ----------
nexttile

scatter(YTest,YPred,35,'filled','MarkerFaceAlpha',0.7)
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
    'FontSize',12,'FontWeight','bold')

%% ---------- (b) REGRESSION FIT ----------
nexttile

p = polyfit(YTest,YPred,1);
yfit = polyval(p,YTest);

scatter(YTest,YPred,30,'filled','MarkerFaceAlpha',0.7)
hold on

plot(YTest,yfit,'k','LineWidth',2)

xlabel('True RET (sec)')
ylabel('Predicted RET (sec)')

title('(b) Regression Fit')

grid on
axis square

text(min(YTest)+2,max(YTest)-5,...
    sprintf('Slope = %.3f',p(1)),...
    'FontSize',12,'FontWeight','bold')

sgtitle('Prediction Performance of CNN Baseline Model')

%% =====================================================
%% FIGURE 2 : ERROR ANALYSIS
%% =====================================================

figure('Color','w','Position',[200 200 900 400]);

tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

%% ---------- (a) RESIDUAL HISTOGRAM ----------
nexttile

histogram(errors,30,'FaceColor',[0.2 0.4 0.8],'EdgeColor','none')

xlabel('Residual Error (sec)')
ylabel('Frequency')

title('(a) Residual Distribution')

grid on

%% ---------- (b) BLAND–ALTMAN ----------
nexttile

mean_vals = (YTest + YPred)/2;
diff_vals = errors;

mean_diff = mean(diff_vals);
std_diff  = std(diff_vals);

upper = mean_diff + 1.96*std_diff;
lower = mean_diff - 1.96*std_diff;

scatter(mean_vals,diff_vals,30,'filled','MarkerFaceAlpha',0.7)
hold on

yline(mean_diff,'k','LineWidth',2)
yline(upper,'r--','LineWidth',2)
yline(lower,'r--','LineWidth',2)

xlabel('Mean RET (sec)')
ylabel('Prediction Difference (sec)')

title('(b) Bland–Altman Analysis')

grid on

text(min(mean_vals)+2,upper-2,...
    sprintf('Bias = %.2f',mean_diff),...
    'FontSize',11,'FontWeight','bold')

sgtitle('Residual and Agreement Analysis')

fprintf('IEEE-style figures generated successfully.\n');