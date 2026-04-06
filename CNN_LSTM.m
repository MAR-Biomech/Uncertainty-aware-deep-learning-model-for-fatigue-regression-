    clc; clear; close all;
    
    fprintf('\n====================================================\n');
    fprintf('      CNN FEATURE + LSTM RET PIPELINE\n');
    fprintf('====================================================\n');
    
    model_file = 'CNN_LSTM_RET_MODEL.mat';
    
    %% =====================================================
    %% LOAD DATA
    %% =====================================================
    fprintf('\nLoading dataset...\n');
    load FINAL_MULTI_MUSCLE_RET_CWT.mat
    
    X = FINAL_DATA.CWT;
    Y = FINAL_DATA.RET(:);
    
    N = size(X,4);
    
    fprintf('Dataset Loaded Successfully.\n');
    fprintf('Total Samples        : %d\n', N);
    fprintf('Input Size           : %d x %d x %d\n', size(X,1), size(X,2), size(X,3));
    fprintf('RET Range            : %.2f to %.2f sec\n', min(Y), max(Y));
    fprintf('Mean RET             : %.2f sec\n', mean(Y));
    fprintf('----------------------------------------------------\n');
    
    %% =====================================================
    %% CHECK IF MODEL EXISTS
    %% =====================================================
    if isfile(model_file)
    
        fprintf('\nPre-trained model found. Loading model...\n');
        load(model_file);
    
        netCNN  = MODEL.CNN;
        netLSTM = MODEL.LSTM;
        seq_len = MODEL.seq_len;
        featureLayer = MODEL.featureLayer;
    
        fprintf('Model loaded successfully!\n');
    
    else
    
        fprintf('\nNo existing model found. Starting training...\n');
    
        %% =====================================================
        %% CNN FEATURE EXTRACTOR
        %% =====================================================
        fprintf('\n--- Training CNN Feature Extractor ---\n');
    
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
    
        fprintf('CNN Train Samples     : %d\n', Ntrain);
        fprintf('CNN Test Samples      : %d\n', N-Ntrain);
    
        XTrainCNN = X(:,:,:,idx(1:Ntrain));
        YTrainCNN = Y(idx(1:Ntrain));
    
        optionsCNN = trainingOptions('adam',...
            'MaxEpochs',15,...
            'MiniBatchSize',32,...
            'Verbose',false);
    
        netCNN = trainNetwork(XTrainCNN,YTrainCNN,layersCNN,optionsCNN);
    
        fprintf('CNN Training Completed.\n');
    
        %% =====================================================
        %% FEATURE EXTRACTION
        %% =====================================================
        fprintf('\n--- Extracting CNN Features ---\n');
        featureLayer = 'relu_feature';
        features = zeros(N,64);
    
        for i = 1:N
            features(i,:) = activations(netCNN,X(:,:,:,i),featureLayer,'OutputAs','rows');
        end
    
        fprintf('Feature Matrix Size   : %d x %d\n', size(features,1), size(features,2));
    
        %% =====================================================
        %% BUILD TEMPORAL SEQUENCES
        %% =====================================================
        fprintf('\n--- Building Temporal Sequences ---\n');
    
        seq_len = 8;
        Xseq = {};
        Yseq = [];
    
        count = 1;
        for i = 1:(N-seq_len)
            Xseq{count} = features(i:i+seq_len-1,:)';
            Yseq(count,1) = Y(i+seq_len-1);
            count = count + 1;
        end
    
        fprintf('Sequence Length        : %d\n', seq_len);
        fprintf('Total Sequences        : %d\n', length(Yseq));
        fprintf('Sequence Feature Size  : %d x %d\n', size(Xseq{1},1), size(Xseq{1},2));
    
        %% =====================================================
        %% TRAIN TEST SPLIT (SEQUENCE)
        %% =====================================================
        rng(1)
        Nseq = length(Yseq);
        idx_seq = randperm(Nseq);
        Ntrain_seq = round(0.8*Nseq);
    
        XTrain = Xseq(idx_seq(1:Ntrain_seq));
        YTrain = Yseq(idx_seq(1:Ntrain_seq));
        XTest  = Xseq(idx_seq(Ntrain_seq+1:end));
        YTest  = Yseq(idx_seq(Ntrain_seq+1:end));
    
        fprintf('LSTM Train Sequences   : %d\n', length(YTrain));
        fprintf('LSTM Test Sequences    : %d\n', length(YTest));
    
        %% =====================================================
        %% LSTM NETWORK
        %% =====================================================
        fprintf('\n--- Training LSTM ---\n');
    
        numFeatures = size(XTrain{1},1);
    
        layersLSTM = [
        sequenceInputLayer(numFeatures)
        lstmLayer(64,'OutputMode','last')
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(1)
        regressionLayer];
    
        optionsLSTM = trainingOptions('adam',...
            'MaxEpochs',40,...
            'MiniBatchSize',16,...
            'Verbose',false);
    
        netLSTM = trainNetwork(XTrain,YTrain,layersLSTM,optionsLSTM);
    
        fprintf('LSTM Training Completed.\n');
    
        %% SAVE MODEL
        MODEL.CNN = netCNN;
        MODEL.LSTM = netLSTM;
        MODEL.seq_len = seq_len;
        MODEL.featureLayer = featureLayer;
        save(model_file,'MODEL','-v7.3');
    
        fprintf('Model saved successfully!\n');
    end
    
    %% =====================================================
    %% EVALUATION
    %% =====================================================
    fprintf('\n--- Evaluating on Test Set ---\n');
    
    % Recreate sequences for full evaluation
    features = zeros(N,64);
    for i = 1:N
        features(i,:) = activations(netCNN,X(:,:,:,i),featureLayer,'OutputAs','rows');
    end
    
    Xseq = {};
    Yseq = [];
    count = 1;
    for i = 1:(N-seq_len)
        Xseq{count} = features(i:i+seq_len-1,:)';
        Yseq(count,1) = Y(i+seq_len-1);
        count = count + 1;
    end
    
    Nseq = length(Yseq);
    rng(1)
    idx_seq = randperm(Nseq);
    Ntrain_seq = round(0.8*Nseq);
    
    XTest  = Xseq(idx_seq(Ntrain_seq+1:end));
    YTest  = Yseq(idx_seq(Ntrain_seq+1:end));
    
    YPred = predict(netLSTM,XTest);
    
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
    
    fprintf('\nPipeline Finished Successfully.\n');
    %%
    %% =====================================================
%% VISUALIZATION SECTION
%% =====================================================

fprintf('\nGenerating evaluation plots...\n');

%% 1️⃣ TRUE vs PREDICTED SCATTER
figure('Name','True vs Predicted RET','Color','w');
scatter(YTest, YPred, 25, 'filled');
hold on
plot([min(YTest) max(YTest)],...
     [min(YTest) max(YTest)],...
     'r','LineWidth',2);
xlabel('True RET (sec)');
ylabel('Predicted RET (sec)');
title(sprintf('CNN+LSTM RET Prediction (R^2 = %.3f)',R2));
grid on


%% 2️⃣ REGRESSION LINE FIT
p = polyfit(YTest, YPred, 1);
yfit = polyval(p, YTest);

figure('Name','Regression Fit','Color','w');
scatter(YTest, YPred, 20, 'filled'); hold on
plot(YTest, yfit, 'k','LineWidth',2);
xlabel('True RET (sec)');
ylabel('Predicted RET (sec)');
title('Linear Fit Between True and Predicted');
grid on


%% 3️⃣ RESIDUAL HISTOGRAM
figure('Name','Residual Distribution','Color','w');
histogram(errors,30);
xlabel('Prediction Error (sec)');
ylabel('Frequency');
title('Residual Histogram');
grid on


%% 4️⃣ RESIDUAL vs TRUE RET
figure('Name','Residual vs True RET','Color','w');
scatter(YTest, errors, 25, 'filled');
yline(0,'r','LineWidth',2);
xlabel('True RET (sec)');
ylabel('Residual (sec)');
title('Residual vs True RET');
grid on


%% 5️⃣ ERROR vs PREDICTED RET
figure('Name','Residual vs Predicted RET','Color','w');
scatter(YPred, errors, 25, 'filled');
yline(0,'r','LineWidth',2);
xlabel('Predicted RET (sec)');
ylabel('Residual (sec)');
title('Residual vs Predicted RET');
grid on


%% 6️⃣ CUMULATIVE ABSOLUTE ERROR DISTRIBUTION
abs_errors = abs(errors);
sorted_err = sort(abs_errors);
cdf_vals = (1:length(sorted_err))/length(sorted_err);

figure('Name','Cumulative Error Distribution','Color','w');
plot(sorted_err, cdf_vals,'LineWidth',2);
xlabel('Absolute Error (sec)');
ylabel('Cumulative Probability');
title('Cumulative Absolute Error');
grid on


%% 7️⃣ BLAND–ALTMAN PLOT (OPTIONAL BUT STRONG)
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

excel_file = 'CNN_LSTM_RET_Results.xlsx';

%% -----------------------------------------------------
%% Prediction Results
%% -----------------------------------------------------

results_table = table;

results_table.True_RET      = YTest;
results_table.Predicted_RET = YPred;
results_table.Error         = errors;
results_table.Abs_Error     = abs(errors);

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
%% Additional Error Statistics
%% -----------------------------------------------------

error_stats = table;

error_stats.Mean_Absolute_Error = mean(abs(errors));
error_stats.Median_Absolute_Error = median(abs(errors));
error_stats.Max_Absolute_Error = max(abs(errors));
error_stats.Min_Absolute_Error = min(abs(errors));

writetable(error_stats,excel_file,'Sheet','Error_Stats');

fprintf('Excel file saved: %s\n',excel_file);
fprintf('================================================\n');
%%
%% =====================================================
%% IEEE-STYLE VISUALIZATION
%% =====================================================

fprintf('\nGenerating publication-quality plots...\n');

set(groot,'defaultAxesFontSize',12);
set(groot,'defaultAxesFontName','Times New Roman');

%% =====================================================
%% FIGURE 1 : PREDICTION PERFORMANCE
%% =====================================================

figure('Color','w','Position',[200 200 900 420]);

tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

%% -------- (a) TRUE vs PREDICTED --------
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
    'FontWeight','bold')

%% -------- (b) REGRESSION FIT --------
nexttile

% Compute regression
p = polyfit(YTest,YPred,1);   % p(1) = slope, p(2) = intercept

% Sort X for smooth line plotting
[YTest_sorted, idx_sort] = sort(YTest);
yfit = polyval(p,YTest_sorted);

scatter(YTest,YPred,35,'filled','MarkerFaceAlpha',0.7)
hold on

plot(YTest_sorted,yfit,'k','LineWidth',2)

xlabel('True RET (sec)')
ylabel('Predicted RET (sec)')
title('(b) Regression Fit')

grid on
axis square

text(min(YTest)+2,max(YTest)-5,...
    sprintf('Slope = %.3f',p(1)),...
    'FontWeight','bold')
sgtitle('CNN-LSTM Prediction Performance')


%% =====================================================
%% FIGURE 2 : ERROR ANALYSIS
%% =====================================================

figure('Color','w','Position',[200 200 900 420]);

tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

%% -------- (a) RESIDUAL HISTOGRAM --------
nexttile

histogram(errors,30,...
    'FaceColor',[0.2 0.4 0.8],...
    'EdgeColor','none')

xlabel('Prediction Error (sec)')
ylabel('Frequency')
title('(a) Residual Distribution')

grid on


%% -------- (b) BLAND–ALTMAN --------
nexttile

mean_vals = (YTest + YPred)/2;
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

text(min(mean_vals)+2,upper-2,...
    sprintf('Bias = %.2f',mean_diff),...
    'FontWeight','bold')

sgtitle('Error and Agreement Analysis')

fprintf('Publication-quality plots generated successfully.\n');