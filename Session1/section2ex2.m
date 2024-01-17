%% Section 2 - exercise 2 setup
% Student numer: 0755676 
% Largest numbers: 7 7 6 6 5
% Descending order: 77665
clear
clc
close all
load("Data_Problem1_regression.mat");

d1=7;
d2=7;
d3=6;
d4=6;
d5=5;

Tnew = (d1*T1 + d2*T2 + d3*T3 + d4*T4 + d5*T5)/(d1 + d2 + d3 + d4 + d5);

TrainingSetX1 = datasample(X1, 1000, 'Replace', false);
ValidationSetX1 = datasample(setdiff(X1, TrainingSetX1), 1000, 'Replace', false);
TestSetX1 = datasample(setdiff(setdiff(X1, TrainingSetX1), ValidationSetX1), 1000, 'Replace', false);

TrainingSetX2 = datasample(X2, 1000, 'Replace', false);
ValidationSetX2 = datasample(setdiff(X2, TrainingSetX2), 1000, 'Replace', false);
TestSetX2 = datasample(setdiff(setdiff(X2, TrainingSetX2), ValidationSetX2), 1000, 'Replace', false);

TrainingSetTnew = datasample(Tnew, 1000, 'Replace', false);
ValidationSetTnew = datasample(setdiff(Tnew, TrainingSetTnew), 1000, 'Replace', false);
TestSetTnew = datasample(setdiff(setdiff(Tnew, TrainingSetTnew), ValidationSetTnew), 1000, 'Replace', false);

%% Visualizing the training set

X1_data = TrainingSetX1;
X2_data = TrainingSetX2;
Y_data = TrainingSetTnew;

% define an interpolation function for your data
f = scatteredInterpolant(X1_data, X2_data, Y_data);

% create a regular grid and evaluate the interpolations on it
x1 = linspace(min(X1_data), max(X1_data), 1000);
x2 = linspace(min(X2_data), max(X2_data), 1000);
[X1_mesh, X2_mesh] = meshgrid(x1, x2);
Y_mesh = f(X1_mesh, X2_mesh);

% Plot a 3D mesh of the grid with its interpolated values
figure;
mesh(X1_mesh, X2_mesh, Y_mesh);
title('Some 3D function');
xlabel('X 1');
ylabel('X 2');
zlabel('Y');

% Optionally add markers to get an idea of how your data is distributed
hold on;
plot3(X1_data, X2_data, Y_data,'.','MarkerSize',15)

%% Build and train your feedforward Neural Network: use the training and validation sets. 
% Build the ANN with 2 inputs and 1 output. Select a suitable model for the 
% problem (number of hidden layers, number of neurons in each hidden layer). 
% Select the learning algorithm and the transfer function that may work 
% best for this problem.

% vergelijk eerst verschillende gradient descent algorithmes en dan
% anderen.

TrainingInput = [TrainingSetX1 TrainingSetX2]';
TrainingTarget = TrainingSetTnew';

% Zie algorim1.m voor voorbeeld hoe dit te doen

alg = 'traincgf';
H = 200; %number of neurons in the hidden layer
delta_epochs = [1,14,985];% Number of epochs to train in each step
epochs = cumsum(delta_epochs);

net = feedforwardnet(H,alg);
net = configure(net, TrainingInput, TrainingTarget);

%net.divideFcn = 'dividetrain';

net = init(net);

net.trainParam.epochs=delta_epochs(1);
net=train(net,TrainingInput,TrainingTarget);
a1=sim(net,TrainingInput);

net.trainParam.epochs=delta_epochs(2);
net=train(net,TrainingInput,TrainingTarget);
a2=sim(net,TrainingInput);

net.trainParam.epochs=delta_epochs(3);
net=train(net,TrainingInput,TrainingTarget);
a3=sim(net,TrainingInput);


% This is badly represented on a figure beceause of 2 inputs?
% how to change the transfer function?
% use the function used earlier

figure
plot(TrainingInput,TrainingTarget,'bx',TrainingInput,a1,'r'); 
title([num2str(epochs(1)),' epochs']);
legend('target',alg1,alg2,'Location','north');





