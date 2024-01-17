clc
clear all;
close all;

%% Load data
% Santa Fe

lag = 50;
layers = [40];
epochs = 50;

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

dataTestStandardized = (dataTest - mu) / sig;

%% Format data -> lag = k with k=1;2;3;...;n
%%only train the LSTM on the lagged data

[XTrainTimeSeries, YTrainTimeSeries] = getTimeSeriesTrainData(dataTrainStandardized', lag);

[XTestTimeSeries, YTestTimeSeries] = getTimeSeriesTrainData(dataTestStandardized', lag);

%% Build ANN

net = feedforwardnet(layers, 'trainlm');
net.numInputs = lag;
net.trainParam.epochs = epochs;
net.divideFcn = 'dividetrain';

%net.divideParam.trainRatio = 0.7;
%net.divideParam.valRatio = 0.15;
%net.divideParam.testRatio = 0.15;
%net.trainParam.max_fail = 100;
for j=1:lag
    net.inputs{j}.size = 1;
    net.inputConnect(1,j) = 1;
end

[net,tr]=train(net,XTrainTimeSeries,YTrainTimeSeries);

%% Test ANN

YPred = YTrainTimeSeries(end);
input = XTrainTimeSeries(:,end);

input = [input(2:end); YPred];
YPred = sim(net, input);
YPredArray=YPred;

%%
for j=2:100
    lastColumn = input(:, end);
    lastColumn = lastColumn(2:end);
    newColumn = [lastColumn; YPred];
    input=[input newColumn];
    YPred = sim(net,input(:,j));%generate new output
    YPredArray(:,j) = YPred; 
end

%%
%YPredNorm = YPredArray*sig+mu;

%compare against dataTest
%rmse = sqrt(mean((YPredNorm-dataTest).^2));
rmse = sqrt(mean(YPredArray-dataTestStandardized).^2);

%%
figure
subplot(2,1,1)
%plot(dataTest)
plot(dataTestStandardized)
hold on
%plot(YPredNorm,'.-')
plot(YPredArray,'.-')
hold off
legend(["Observed" "Forecast"])
title("Forecast")

subplot(2,1,2)
%stem(YPredNorm - dataTest)
stem(YPredArray - dataTestStandardized)
xlabel("Time Steps")
ylabel("Error")
title("RMSE = " + rmse)
%%
figure
plot(dataTrainStandardized)
hold on
idx = 1000:(1000+100);
%plot(idx,[data(1000) YPredNorm],'.-')
plot(idx,[dataTrainStandardized(1000) YPredArray],'.-')
hold off
xlabel("Time Steps")
ylabel("Data")
title("Forecast")
legend(["Observed" "Forecast"])


