close all;
clear;
clc;

%% First show a figure of the training and test set

% NOT NORMALIZED

data_train = readtable("lasertrain.dat");
data_train(:,1:3) = [];
data_train = table2array(data_train)';

data_test = readtable("laserpred.dat");
data_test(:,1:3) = [];
data_test = table2array(data_test)';

%data = cat(2, data_train, data_test);

% figure;
% hold on;
% plot(data_train, 'b') % Plotting training data in blue
% plot(1000:size(data_test, 2) + 999, data_test, 'r') % Plotting test data in red starting from x-axis value 1000
% xlim([0, 1100]);
% xlabel('Discrete time');
% hold off;

%% Showing the training set and test set
% NORMALIZED:

muTrain = mean(data_train);
sigTrain = std(data_train);

dataTrainStandardized = (data_train - muTrain) / sigTrain;

muTest = mean(data_test);
sigTest = std(data_test);

dataTestStandardized = (data_test - muTest) / sigTest;

% figure;
% hold on;
% plot(dataTrainStandardized, 'b') % Plotting training data in blue
% plot(1000:size(dataTestStandardized, 2) + 999, dataTestStandardized, 'r') % Plotting test data in red starting from x-axis value 1000
% xlabel('Discrete time k');
% xlim([0, 1100]);
% xlabel('Discrete time');
% hold off;

%%
dataTrainStandardized = dataTrainStandardized';
dataTestStandardized = dataTestStandardized';

%% testing - toy example
p = 20;
[training, target_train] = getTimeSeriesTrainData(dataTrainStandardized, p);

%% Training ANN in feed forward

% Getting target and training data with lag
max_lag = 100;
H = 10;
alg = 'trainlm';
errors = zeros(1,max_lag);

for i=1:max_lag
    fprintf('Starting iteration %s of %s.', i, max_lag);
    
    [training, target_train] = getTimeSeriesTrainData(dataTrainStandardized, i);
    
    % Formatting training data for train function
    train_set = cell(i,1);
    
    for j=1:i
        train_set{j,1} = training(j,:);
    end
    
    % Configuring and training the network
    net = feedforwardnet(H, alg);
    net.numInputs = i;
    net.trainParam.epochs = 100;
    net.divideFcn = 'dividerand';
    %net.divideParam.trainRatio = 0.7;
    %net.divideParam.valRatio = 0.15;
    %net.divideParam.testRatio = 0.15;
    %net.trainParam.max_fail = 100;
    for j=1:i
        net.inputs{j}.size = 1;
        net.inputConnect(1,j) = 1;
    end
    
    [net,tr] = train(net, train_set, target_train);
    
    % Getting target and test data with lag
    [test, target_predict] = getTimeSeriesTrainData(dataTestStandardized, i);
    
    prediction = zeros(1,100-i);
    x = test(:,1);
    
    for j=1:100-i
        y = net(x);
        prediction(1, j) = y;
        x = vertcat(x(2:end),y);
    end
    
    error = sqrt(mean((prediction - target_predict).^2));
    errors(i) = error;
end

%% Plotting error

figure;
hold on;
plot(errors);
xlabel('Lag');
ylabel('RMSE');
hold off;

%% Prediction on test set
figure 
hold on;
plot(prediction);
plot(target_predict);
legend('prediction','target');
hold off;

%% RMSE in function of lag with fixed amount of hidden neurons

max_lag = 100;
errors = zeros(1,max_lag);
H = 40;
alg = 'trainlm';

for i=1:max_lag
    
    [training, target_train] = getTimeSeriesTrainData(dataTrainStandardized, i);
    
    % Formatting training data for train function
    train_set = cell(i,1);
    
    for j=1:i
        train_set{j,1} = training(j,:);
    end
    
    % Configuring and training the network
    net = feedforwardnet(H, alg);
    net.numInputs = i;
    net.trainParam.epochs = 100;
    net.divideFcn = 'dividerand';
    %net.divideParam.trainRatio = 0.7;
    %net.divideParam.valRatio = 0.15;
    %net.divideParam.testRatio = 0.15;
    %net.trainParam.max_fail = 100;
    for j=1:i
        net.inputs{j}.size = 1;
        net.inputConnect(1,j) = 1;
    end
    
    [net,tr] = train(net, train_set, target_train);
    
    % Getting target and test data with lag
    [test, target_predict] = getTimeSeriesTrainData(dataTestStandardized, i);
    
    prediction = zeros(1,100-i);
    x = test(:,1);

    
    for j=1:100-i
        y = net(x);
        prediction(1, j) = y;
        x = vertcat(x(2:end),y);
    end
    
    error = sqrt(mean((prediction - target_predict).^2));
    errors(i) = error;
end

%%
plot(errors)
xlabel('Lag (p)') 
ylabel('RMSE') 
%title(alg)

%% Test amount of hidden units versus lag

lag_list= [5 ; 10 ; 15; 20 ; 25 ; 30 ;40 ; 50 ; 60]';
H_list = [5;10;20;30;40;50;60;70;80]';
alg = 'trainlm';                                                            % HIER OOK MSS ANDERE ALG proberen?
%alg= 'trainscg';

epochs = 100;

errors = zeros(length(lag_list),length(H_list));

for indexH=1:length(H_list)
    H = H_list(indexH);
    fprintf('Starting iteration %f.\n', indexH);

    for index=1:length(lag_list)
        i = lag_list(index);
        [training, target_train] = getTimeSeriesTrainData(dataTrainStandardized, i);
    
        % Formatting training data for train function
        train_set = cell(i,1);
        
        for j=1:i
            train_set{j,1} = training(j,:);
        end
        
        % Configuring and training the network
        net = feedforwardnet(H, alg);
        net.numInputs = i;
        net.trainParam.epochs = epochs;
        net.divideFcn = 'dividerand';
        %net.divideParam.trainRatio = 0.7;
        %net.divideParam.valRatio = 0.15;
        %net.divideParam.testRatio = 0.15;
        %net.trainParam.max_fail = 100;
        for j=1:i
            net.inputs{j}.size = 1;
            net.inputConnect(1,j) = 1;
        end
        
        [net,tr] = train(net, train_set, target_train);
        
        % Getting target and test data with lag
        [test, target_predict] = getTimeSeriesTrainData(dataTestStandardized, i);
        
        prediction = zeros(1,100-i);
        x = test(:,1);
        
        for j=1:100-i
            y = net(x);
            prediction(1, j) = y;
            x = vertcat(x(2:end),y);
            %x = [x(:, 2:end), y];
        end
        
        error = sqrt(mean((prediction - target_predict).^2));     %Dit is de error op de test set
        errors(index, indexH) = error;
    end
end

%%
% Create grid of H and lag values
[H_vals, lag_vals] = meshgrid(H_list, lag_list);

% Create 3D plot
figure;
plot3(H_vals(:), lag_vals(:), errors(:));
xlabel('H');
ylabel('Lag');
zlabel('RMSE');
title('Error Surface');
xlim([5, 80]);
ylim([5, 60]);
%set(gca, 'ZScale', 'log');  % Set z-axis to logarithmic scale

%%
figure;
hold on;
color_map = lines(length(lag_list));
for index = 1:length(lag_list)
    plot3(H_vals(:, index), lag_vals(:, index), errors(:, index), 'o-', 'Color', color_map(index, :));
end
hold off;
xlabel('Amount of hidden units');
ylabel('Lag (p)');
zlabel('RMSE');
xlim([5, 80]);
ylim([5, 60]);

%% nu verder werken en een ANN bouwen predictions tonen

%Met train lm, hidden units = 50 nemen en P = 50
%trainscg was ook getest, deze trainde sneller maar de RMSE lag hoger

alg = 'trainlm';  
H = 50;
i = 50; %The amount of lag and amount of inputs to the NN

[training, target_train] = getTimeSeriesTrainData(dataTrainStandardized, i);
    
% Formatting training data for train function
train_set = cell(i,1);
     
for j=1:i
    train_set{j,1} = training(j,:);
end
        
% Configuring and training the network
net = feedforwardnet(H, alg);
net.numInputs = i;
net.trainParam.epochs = 100;
net.divideFcn = 'dividerand';
for j=1:i
    net.inputs{j}.size = 1;
    net.inputConnect(1,j) = 1;
end
        
[net,tr] = train(net, train_set, target_train);
        
% Getting target and test data with lag
[test, target_predict] = getTimeSeriesTrainData(dataTestStandardized, i);
        
prediction = zeros(1,100-i);
x = test(:,1);
        
for j=1:100-i
    y = net(x);
    prediction(1, j) = y;
    x = vertcat(x(2:end),y);
    %x = [x(:, 2:end), y];
end
        
error = sqrt(mean((prediction - target_predict).^2));     %Dit is de error op de test set
%%
fprintf('The RMSE for algorithm %s with %.3f hidden units is: %.3f\n', alg, H, error);

%%
figure; %Figuur van de prediction op de test set
hold on;
plot(dataTestStandardized,'DisplayName','Test set');
plot(prediction,'DisplayName','Prediction');
legend('Test set','Prediction','Location','northeast');
hold off;


%%  Op test set tonen

close all;
clear;
clc;

p = 30;
H = 35;
alg = 'trainlm';

train_set = load('lasertrain.dat');
train_mean = mean(train_set);
train_std = std(train_set);

train_set = (train_set - train_mean)/train_std;

X_train = getTimeSeriesTrainData(train_set,p);

% Build ANN
net = feedforwardnet(H, alg); % net with 1 hidden layers

net.trainParam.epochs = 1000;
net.divideFcn = 'divideblock';

input = X_train(1:p-1,:)';
target = X_train(p,:)';

net = train(net, input' , target');

% Predict future
test_set = load('laserpred.dat');

test_set = (test_set - train_mean)/train_std;

X_test = getTimeSeriesTrainData(test_set,p);
%y_test = sim(net, X_train(1:p-1, end-p+1:end));
y_test = X_train(p, end-p+1:end);
y_test = [y_test sim(net, X_train(1:p-1, end-1))];
%y_test = [y_test sim(net,X_test(1:p-1,:))];

e = gsubtract(y_test(2:end), test_set(1:end-1)');
rmse = sqrt(mse(e));

fprintf('The RMSE is: %f.3',rmse);
figure;
hold on;
plot(test_set,'b');
plot(y_test,'r');
legend('Target','Predicted','Location','northeast');
xlabel('Discrete time');
hold off;


%%
%predict_set = sim(net, X_train(1:p-1, end-p+1:end));
predict_set = X_train(p, end-p+1:end);

for i=1:100
    predict_set(p+i) = sim(net, predict_set(i+1:p+i-1)');
end

e = gsubtract(predict_set(p+2:end), test_set(1:end-1)');
rmse = sqrt(mse(e));

fprintf('The MSE for the next 100 predictions after the training set are: %f.3',rmse);
figure;
hold on;
plot(test_set,'b');
plot(predict_set(p+2:end),'r')
legend('Target','Predicted','Location','northeast');
xlabel('Discrete time');
%ylabel('');
hold off;

