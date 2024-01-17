%% Bayesian

clc;
clear;
close all;

alg1 = 'trainlm';
alg2 = 'trainbr';

H = 400;% Number of neurons in the hidden layer  
epochs = 100;

%generation of examples and targets
dx=0.05;% Decrease this value to increase the number of data points
x=0:dx:3*pi;y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=yn;% Targets. Change to yn to train on noisy data

net1=feedforwardnet(H,alg1);% Define the feedfoward net (hidden layers)
net2=feedforwardnet(H,alg2);

net1=configure(net1,x,t);% Set the input and output sizes of the net
net2=configure(net2,x,t);

%net1.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
%net2.divideFcn = 'dividetrain';

net1=init(net1);% Initialize the weights (randomly)
net2.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};

a1 = net1.iw{1,1};
a2 = net1.lw{2,1};
a3 = net1.b{1};
a4 = net1.b{2};

net1.trainParam.epochs=epochs;  % set the number of epochs for the training 
net2.trainParam.epochs=epochs;

net1.trainParam.max_fail = 10;
net2.trainParam.max_fail = 10;

%%
[net1,tr1]=train(net1,x,t);   % train the networks

%%
[net2,tr2]=train(net2,x,t);

%%
b1 = net2.iw{1,1};
b2 = net2.lw{2,1};
b3 = net2.b{1};
b4 = net2.b{2};

%%
a13=sim(net1,x);  % simulate the networks with the input vector x
a23=sim(net2,x);  

%% figure MSE
figure;
hold on;
%plot(tr{i}.perf,'DisplayName',nets{i}.trainFcn,'LineWidth',1.5);
plot(tr1.perf,'DisplayName',net1.trainFcn,'LineWidth',1.5);
plot(tr2.perf,'DisplayName',net2.trainFcn,'LineWidth',1.5);
legend(alg1,alg2,'Location','northeast');
set(gca, 'YScale', 'log')
xlabel('Epochs');
ylabel('MSE');
hold off;

%% figure sine wave vs prediction
figure;
hold on;
plot(x,t,'bx',x,a13,'g',x,a23,'r'); % plot the sine function and the output of the networks
title(num2str(epochs),' epochs');
legend('target',alg1,alg2,'Location','northwest');
hold off;

%% generate tabel -> MSE ? When does it converge? Effective number of paramters?
% Met fprintf doen
%TODO


