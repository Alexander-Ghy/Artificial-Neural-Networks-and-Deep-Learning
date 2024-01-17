close all;
clear;
clc;


train_set = load('lasertrain.dat');
train_mean = mean(train_set);
train_std = std(train_set);

train_set = (train_set - train_mean)/train_std;
%%
p = 30;
X_train = getTimeSeriesTrainData(train_set,p);
H = 50;
algorithm = 'trainlm';


%% Build ANN

net = feedforwardnet(H, algorithm); % net with 1 hidden layers

net.trainParam.epochs = 200;
net.divideFcn = 'divideblock';
net.trainParam.max_fail = 3;
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio   = 0.2;
net.divideParam.testRatio  = 0;

input = X_train(1:p-1,:)';
target = X_train(p,:)';

net = train(net, input' , target');

%% Predict original training set
clf

y_train = sim(net, X_train(1:p-1,:))
plot(y_train,'r')
hold on
plot(X_train(p,:),'b')


%% Predict future



test_set = load('laserpred.dat');

test_set = (test_set - train_mean)/train_std;

X_test = getTimeSeriesTrainData(test_set,p);
%% was gecomment

y_test = sim(net, X_train(1:p-1, end-p+1:end));
%%
%y_test = X_train(p, end-p+1:end);

%% was gecomment

y_test2 = sim(net, X_train(1:p-1, end-1))

%%
y_test = [y_test sim(net,X_test(1:p-1,:))];

%%
figure;
plot(y_test,'r')
hold on
plot(test_set,'b')
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

e = e.^2;
e = mean(e);

rmse = sqrt(e)
plot(predict_set(p+2:end),'r')
legend('Target','Predicted')