clear;
clc;

% Santa Fe
load("lasertrain.dat");
% Normalization
trainingMean = mean(lasertrain);
trainingStd = sqrt(mean((lasertrain - trainingMean).^2));
lasertrain = (lasertrain-trainingMean)/trainingStd;

load("laserpred.dat");
% Normalization with training parameters to avoid data snooping
laserpred = (laserpred - trainingMean)/trainingStd;

p = 50;
[trainingX,trainingY]=getTimeSeriesTrainData(lasertrain, p);
[testX,testY]=getTimeSeriesTrainData([lasertrain;laserpred],p);
testX = testX(:,end-99:end);
testY = testY(end-99:end);
trainingP = con2seq(trainingX);
trainingT = con2seq(trainingY);
testP = con2seq(testX);
testT = con2seq(testY);
 
% Plot MSE vs HiddenUnits in training and validation
maxHidden = 40;
for i=40:maxHidden
    nets{i} = feedforwardnet(i,'trainbr');
    nets{i}.trainParam.epochs=200;
    [nets{i},tr{i}]=train(nets{i},trainingP,trainingT);
    simulationTraining{i}=sim(nets{i},trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    
    % Closed loop test simulation
    mseValidation{i} = 0;
    lastOutput = trainingY(end);
    input = trainingX(:,end);
    for j=1:length(testY)
        input = [input(2:end);lastOutput];
        lastOutput = sim(nets{i},input);
        simulationValidation{i}{j} = lastOutput;
        mseValidation{i} = mseValidation{i} + (lastOutput-testY(j))^2;
    end
    mseValidation{i} = mseValidation{i}/length(testY);
end

figure;
plot(cell2mat(mseTraining),'DisplayName','Training');
hold on;
plot(cell2mat(mseValidation),'DisplayName','Validation');
hold off;
set(gca, 'YScale', 'log')
legend('Training Set', 'Test Set');
xlabel('Number of Hidden Units');
ylabel('MSE');
title('MSE vs Number of Hidden Units')

%%

% Plot results
chosenNet = 40;
%postregm(cell2mat(sim(nets{chosenNet},trainingP)),trainingY);
%postregm(cell2mat(sim(nets{chosenNet},testP)),testY);

figure;
plot(trainingY,'DisplayName','Training set');
hold on;
plot(cell2mat(simulationTraining{chosenNet}),'DisplayName','NN');
hold off;

figure;
plot(testY,'DisplayName','Test set');
hold on;
plot(cell2mat(simulationValidation{chosenNet}),'DisplayName','NN');
legend('Test Set', 'NN approximation')
title('p=50, 40 hidden units')
hold off;