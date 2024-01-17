close all;
clear;
clc;


train_set = load('lasertrain.dat');
train_mean = mean(train_set);
train_std = std(train_set);

train_set = (train_set - train_mean)/train_std;

p = 50;                                                                     %Hier P aanpassen en het verschil bekijken
X_train = getTimeSeriesTrainData(train_set,p);

%% Algorithme bouwen  en trainen

alg = 'trainlm';
H = 50;                                                                     %kijken hoeveel hidden neurons nodig zijn

net = feedforwardnet(H, alg); % net with 1 hidden layers

net.trainParam.epochs = 100;
net.divideFcn = 'divideblock';
net.trainParam.max_fail = 3;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;

input = X_train(1:p-1,:)';                                                  % Training is done in feedforward
target = X_train(p,:)';
net = train(net, input' , target');

%%
test_set = load('laserpred.dat');

test_set = (test_set - train_mean)/train_std;

X_test = getTimeSeriesTrainData(test_set,p);
%y_test = sim(net, X_train(1:p-1, end-p+1:end));
y_test = X_train(p, end-p+1:end);
%y_test2 = sim(net, X_train(1:p-1, end-1))
y_test = [y_test sim(net,X_test(1:p-1,:))];
plot(y_test,'r');
hold on
plot(test_set,'b');
e = gsubtract(y_test(2:end), test_set(1:end-1)');
rmse = sqrt(mse(e))

%%
figure;

plot(test_set)
hold on
%predict_set = sim(net, X_train(1:p-1, end-p+1:end));
predict_set = X_train(p, end-p+1:end);

for i=1:100
    predict_set(p+i) = sim(net, predict_set(i+1:p+i-1)');
end

e = gsubtract(predict_set(p+2:end), test_set(1:end-1)');
rmse = sqrt(mse(e))
plot(predict_set(p+2:end),'r')
legend('Target','Predicted')

