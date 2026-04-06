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

    %% SAVE EVERYTHING
    UNC_MODEL.CNN = netCNN;
    UNC_MODEL.LSTM = netLSTM;
    UNC_MODEL.seq_len = seq_len;
    UNC_MODEL.idx_seq = idx_seq;
    UNC_MODEL.Ntrain_seq = Ntrain_seq;

    save(model_file,'UNC_MODEL','-v7.3');
    fprintf('Model saved successfully.\n');
end

%% =====================================================
%% REBUILD TEST DATA (ALWAYS SAFE)
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
%% UNCERTAINTY
%% =====================================================

p90 = prctile(preds,90,2);
p10 = prctile(preds,10,2);
interval_width = p90 - p10;

sigma_pred = interval_width ./ (abs(mu_pred)+1e-6);
alpha = mean(abs(mu_pred - YTest)) / mean(sigma_pred);
sigma_pred = alpha * sigma_pred;

%% =====================================================
%% METRICS
%% =====================================================

errors = mu_pred - YTest;

RMSE = sqrt(mean(errors.^2));
MAE  = mean(abs(errors));
R2   = 1 - sum(errors.^2)/sum((YTest-mean(YTest)).^2);
r    = corr(YTest,mu_pred);

fprintf('\n================ PERFORMANCE =================\n');
fprintf('RMSE  : %.4f sec\n',RMSE);
fprintf('MAE   : %.4f sec\n',MAE);
fprintf('R^2   : %.4f\n',R2);
fprintf('Corr  : %.4f\n',r);
fprintf('Residual Mean : %.4f\n',mean(errors));
fprintf('Residual Std  : %.4f\n',std(errors));
fprintf('================================================\n');

%% =====================================================
%% VISUALIZATION
%% =====================================================

figure; scatter(YTest,mu_pred,25,'filled'); hold on
plot([min(YTest) max(YTest)],[min(YTest) max(YTest)],'r','LineWidth',2);
title(sprintf('Prediction (R^2=%.3f)',R2));
xlabel('True RET'); ylabel('Predicted RET'); grid on

figure; scatter(sigma_pred,abs(errors),25,'filled');
xlabel('Uncertainty'); ylabel('Absolute Error');
title('Uncertainty–Error Relationship'); grid on

figure; histogram(errors,30);
title('Residual Distribution'); grid on

fprintf('\nPipeline Completed Successfully.\n');