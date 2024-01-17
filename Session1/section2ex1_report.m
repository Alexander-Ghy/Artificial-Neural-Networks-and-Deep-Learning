clear;
clc;
close all;

%Pre-configuration

alg1='traingd';     %gradient descent
alg2='traingda';    %gradient descent with adaptive learning rate
alg3='traingdm';    %gradient descent with momentum
alg4='traingdx';    %gradient descent with momentum and adaptive learning rate
alg5='traincgf';    %Fletcher-Reeves conjugate gradient algorithm
alg6='traincgp';    %Polak-Ribiere conjugate gradient algorithm
alg7='trainbfg';    %BFGS quasi Newton algorithm (quasi Newton)
alg8='trainlm';     %Levenberg-Marquardt algorithm (adaptive mixture of Newton and steepest descent algorithms)
alg9="trainbr";     %Bayesian

H = 50;% Number of neurons in the hidden layer                              %%TODO: deze hier ook mee spelen om te kijken hoeveel minimum nodig zijn, 28?
epochs = 1000;

%generation of examples and targets
dx=0.05;% Decrease this value to increase the number of data points
x=0:dx:3*pi;y=sin(x.^2);
sigma=0.2;% Standard deviation of added noise
yn=y+sigma*randn(size(y));% Add gaussian noise
t=yn;% Targets. Change to yn to train on noisy data

%% verschillende algorithmes vergelijken

net1=feedforwardnet(H,alg1);% Define the feedfoward net (hidden layers)
net2=feedforwardnet(H,alg2);
net3=feedforwardnet(H,alg3);
net4=feedforwardnet(H,alg4);
net5=feedforwardnet(H,alg5);
net6=feedforwardnet(H,alg6);
net7=feedforwardnet(H,alg7);
net8=feedforwardnet(H,alg8);
net9=feedforwardnet(H,alg9);

net1=configure(net1,x,t);% Set the input and output sizes of the net
net2=configure(net2,x,t);
net3=configure(net3,x,t);
net4=configure(net4,x,t);
net5=configure(net5,x,t);
net6=configure(net6,x,t);
net7=configure(net7,x,t);
net8=configure(net8,x,t);
net9=configure(net9,x,t);

net1.divideFcn = 'dividetrain';% Use training set only (no validation and test split)
net2.divideFcn = 'dividetrain';
net3.divideFcn = 'dividetrain';
net4.divideFcn = 'dividetrain';
net5.divideFcn = 'dividetrain';
net6.divideFcn = 'dividetrain';
net7.divideFcn = 'dividetrain';
net8.divideFcn = 'dividetrain';
net9.divideFcn = 'dividetrain';

net1=init(net1);% Initialize the weights (randomly)
net2.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};
net3.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net3.lw{2,1}=net1.lw{2,1};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};
net4.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net4.lw{2,1}=net1.lw{2,1};
net4.b{1}=net1.b{1};
net4.b{2}=net1.b{2};
net5.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net5.lw{2,1}=net1.lw{2,1};
net5.b{1}=net1.b{1};
net5.b{2}=net1.b{2};
net6.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net6.lw{2,1}=net1.lw{2,1};
net6.b{1}=net1.b{1};
net6.b{2}=net1.b{2};
net7.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net7.lw{2,1}=net1.lw{2,1};
net7.b{1}=net1.b{1};
net7.b{2}=net1.b{2};
net8.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net8.lw{2,1}=net1.lw{2,1};
net8.b{1}=net1.b{1};
net8.b{2}=net1.b{2};
net9.iw{1,1}=net1.iw{1,1};% Set the same weights and biases for the networks 
net9.lw{2,1}=net1.lw{2,1};
net9.b{1}=net1.b{1};
net9.b{2}=net1.b{2};

net1.trainParam.epochs=epochs;  % set the number of epochs for the training 
net2.trainParam.epochs=epochs;
net3.trainParam.epochs=epochs;
net4.trainParam.epochs=epochs;
net5.trainParam.epochs=epochs;
net6.trainParam.epochs=epochs;
net7.trainParam.epochs=epochs;
net8.trainParam.epochs=epochs;
net9.trainParam.epochs=epochs;
[net1,tr1]=train(net1,x,t);   % train the networks
[net2,tr2]=train(net2,x,t);
[net3,tr3]=train(net3,x,t);
[net4,tr4]=train(net4,x,t);
[net5,tr5]=train(net5,x,t);
[net6,tr6]=train(net6,x,t);
[net7,tr7]=train(net7,x,t);
[net8,tr8]=train(net8,x,t);
[net9,tr9]=train(net9,x,t);
a13=sim(net1,x);  % simulate the networks with the input vector x
a23=sim(net2,x);  
a33=sim(net3,x);
a43=sim(net4,x);
a53=sim(net5,x);
a63=sim(net6,x);
a73=sim(net7,x);
a83=sim(net8,x);
a93=sim(net9,x);

%%
wb = getwb(net8);

% Calculate the number of parameters
numParameters = length(wb)

%% plot MSE over epochs
figure;
hold on;
%plot(tr{i}.perf,'DisplayName',nets{i}.trainFcn,'LineWidth',1.5);
plot(tr1.perf,'DisplayName',net1.trainFcn,'LineWidth',1.5);
plot(tr2.perf,'DisplayName',net2.trainFcn,'LineWidth',1.5);
plot(tr3.perf,'DisplayName',net3.trainFcn,'LineWidth',1.5);
plot(tr4.perf,'DisplayName',net4.trainFcn,'LineWidth',1.5);
plot(tr5.perf,'DisplayName',net5.trainFcn,'LineWidth',1.5);
plot(tr6.perf,'DisplayName',net6.trainFcn,'LineWidth',1.5);
plot(tr7.perf,'DisplayName',net7.trainFcn,'LineWidth',1.5);
plot(tr8.perf,'DisplayName',net8.trainFcn,'LineWidth',1.5);
plot(tr9.perf,'DisplayName',net9.trainFcn,'LineWidth',1.5);
legend(alg1,alg2,alg3,alg4,alg5,alg6,alg7,alg8,alg9,'Location','northeast');
set(gca, 'YScale', 'log')
xlabel('Epochs');
ylabel('MSE');
title(num2str(epochs), ' epochs');
hold off;

%% Time of the different algorithms in training time.
disp('Alg1:')
tr1.time(end)

disp('Alg2:')
tr2.time(end)

disp('Alg3:')
tr3.time(end)

disp('Alg4:')
tr4.time(end)

disp('Alg5:')
tr5.time(end)

disp('Alg6:')
tr6.time(end)

disp('Alg7:')
tr7.time(end)

disp('Alg8:')
tr8.time(end)

disp('Alg8:')
tr9.time(end)

%% 
figure;
hold on;
plot(x,t,'bx',x,a13,'g',x,a83,'r'); % plot the sine function and the output of the networks
title([num2str(epochs),' epochs']);
legend('target',alg1,alg8,'Location','northwest');
hold off;

%% Alle algorithmes printen op 1 figuur
figure;
hold on;
plot(x,t,'bx',x,a13,x,a23,x,a33,x,a43,x,a53,x,a63,x,a73,x,a83);
title([num2str(epochs),' epochs']);
legend('target',alg1,alg2,alg3,alg4,alg5,alg6,alg7,alg8,'Location','northwest');
hold off;

%% Make table for each algorithm to show: MSE - Linear fit - Converge - # number of parameters

trList = {tr1 , tr2 , tr3 , tr4 , tr5 , tr6 , tr7 , tr8};
aList = {a13, a23, a33, a43, a53, a63, a73 , a83,};
netList = {net1, net2, net3, net4, net5, net6, net7, net8};


for i=1:numel(trList)
    trTemp = trList{i};
    fprintf('\tFollowing information is for algorithm: %s\n', trTemp.trainFcn);
    fprintf('Training time needed: %s\n', trTemp.time(end));
    fprintf('Best MSE: %s\n', trTemp.best_perf); 
    [m,b,r] = postreg(aList{i},y,'hide');
    fprintf('Linear fit: %s\n', r);
    fprintf('Best epoch: %.0f\n', trTemp.best_epoch);
    
    fprintf('\n \t--------- \n\n')
end


%% Figures to show linear fit
figure;
postregm(a13,y);

figure;
postregm(a83,y);

%% Finding MSE in function of amount of hidden units

% Define the number of hidden units to test
%hidden_units = [5 10 15 20 25];
hidden_units = 1:1:100;

% HIER MAX van 1000 EPOCHS

% Initialize the MSE vector
MSE1 = zeros(length(hidden_units),1);
MSE4 = zeros(length(hidden_units),1);
MSE7 = zeros(length(hidden_units),1);
MSE8 = zeros(length(hidden_units),1);

% Train the network for each number of hidden units
for i = 1:length(hidden_units)
    disp("Starting iteration: " + i) 
    % Create the network with the specified number of hidden units
    net1 = feedforwardnet(hidden_units(i), alg1);
    net4 = feedforwardnet(hidden_units(i), alg4);
    net7 = feedforwardnet(hidden_units(i), alg7);
    net8 = feedforwardnet(hidden_units(i), alg8);
    
    % Train the network using the algorithm
    net1.trainFcn = alg1;
    [net1, tr1] = train(net1, x, t);

    net4.trainFcn = alg4;
    [net4, tr4] = train(net4, x, t);

    net7.trainFcn = alg7;
    [net7, tr7] = train(net7, x, t);

    net8.trainFcn = alg8;
    [net8, tr8] = train(net8, x, t);
    
    % Evaluate the performance of the network on the training data
    y1 = net1(x);
    e1 = gsubtract(t, y1);
    MSE1(i) = perform(net1, t, y1);

    y4 = net4(x);
    e4 = gsubtract(t, y4);
    MSE4(i) = perform(net4, t, y4);

    y7 = net7(x);
    e7 = gsubtract(t, y7);
    MSE7(i) = perform(net7, t, y7);

    y8 = net8(x);
    e8 = gsubtract(t, y8);
    MSE8(i) = perform(net8, t, y8);
    

end

% Plot the MSE as a function of the number of hidden units

%% Plot MSE over function of hidden units
figure;
hold on;
plot(hidden_units, MSE1);
plot(hidden_units, MSE4);
plot(hidden_units, MSE7);
plot(hidden_units, MSE8);
legend(alg1,alg4,alg7,alg8,'Location','northeast');
xlabel('Number of Hidden Units');
ylabel('MSE');
set(gca, 'YScale', 'log')
hold off;







