clc
clear all;
close all;

%% Load data
% Santa Fe
load("lasertrain.dat");

load("laserpred.dat");

%%append data to each other
data = [lasertrain.', laserpred.'];

% figure
% plot(data)
% xlabel("Time")
% ylabel("Data")
% title("Santa Fe")

%% Format data -> lag = 1;
numTimeStepsTrain = 1000;

dataTrain = data(1:numTimeStepsTrain);
dataTest = data(numTimeStepsTrain+1:end);

%% Normalize data
mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

%%Split up data into 1000 training samples and 100 predictor samples
XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

%% Format data -> lag = k with k=1;2;3;...;n
%%only train the LSTM on the lagged data

lasertrainNorm = (lasertrain-mu)/sig;

lag = 1;
[XTrainTimeSeries, YTrainTimeSeries] = getTimeSeriesTrainData(dataTrainStandardized', lag);

laserpredNorm = (laserpred-mu)/sig;
[XTestTimeSeries, YTestTimeSeries] = getTimeSeriesTrainData(laserpredNorm, lag);

%% Numfeatures should be equal to the amount of lag introduced into the time series data
numFeatures = lag;
numResponses = 1;
numHiddenUnits = 200;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',500, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');
%% Train the network on the time series data
net = trainNetwork(XTrainTimeSeries,YTrainTimeSeries,layers,options);

%% Make predictions using the trained net
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized;

net = predictAndUpdateState(net, XTrainTimeSeries);

input = XTrainTimeSeries(:,end);
input = [input(2:end); YTrainTimeSeries(end)]; %replace last item of row vector, when lag is one -> should replace value
[net, YPred] = predictAndUpdateState(net,input);
YPredArray = YPred;
%MATLab gives error about dimension of YTrain. It is logical that it does
%so, as it expects 20 input features, but YTrain doesn't have this amount
%of features and it doesn't make sense that it would have 20 input features
%I am not sure how I should format YTrain in order to use it to make
%predictions

%%


numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    lastColumn = input(:, end);
    lastColumn = lastColumn(2:end);
    newColumn = [lastColumn; YPred];
    input=[input newColumn];
    [net,YPred] = predictAndUpdateState(net,input(:, i),'ExecutionEnvironment','cpu');
    YPredArray(:, i) = YPred;
end

%%

YPredArray = sig*YPredArray + mu;

YTest = dataTest;
rmse = sqrt(mean((YPredArray-YTest).^2));

%%
figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPredArray],'.-')
hold off
xlabel("Discrete time")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPredArray,'.-')
hold off
legend(["Observed" "Forecast"])
title("Forecast")

subplot(2,1,2)
stem(YPredArray - YTest)
xlabel("Discrete time")
ylabel("Error")
title("RMSE = " + rmse)




