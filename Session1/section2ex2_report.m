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

% TODO AANPASSEN, ZEKER ZIJN DAT ER NIET IETS DUBBEL ZIT IN TRAINING of al
% iets van training in validation steekt fzo

%Training set
temp = datasample([X1 X2 Tnew],1000,1);
trainingX = temp(:,1:2).';
trainingY = temp(:,3).';
trainingP = con2seq(trainingX);
trainingT = con2seq(trainingY);

%Validation set
temp = datasample([X1 X2 Tnew],1000,1);
validationX = temp(:,1:2).';
validationY = temp(:,3).';
validationP = con2seq(validationX);
validationT = con2seq(validationY);
%Test set
temp = datasample([X1 X2 Tnew],1000,1);
testX = temp(:,1:2).';
testY = temp(:,3).';
testP = con2seq(testX);
testT = con2seq(testY);

% indexTX1 = randperm(length(X1), 1000);
% indexTX2 = randperm(length(X2), 1000);
% indexTTN = randperm(length(Tnew), 1000);
% 
% indexVX1 = randperm(length(X1), 1000);
% indexVX2 = randperm(length(X2), 1000);
% indexVTN = randperm(length(Tnew), 1000);
% 
% indexTEX1 = randperm(length(X1), 1000);
% indexTEX2 = randperm(length(X2), 1000);
% indexTETN = randperm(length(Tnew), 1000);
% 
% TrainingX1 = X1(indexTX1);
% TrainingX2 = X2(indexTX2);
% TrainingTnew = Tnew(indexTTN);
% 
% ValidationX1 = X1(indexVX1);
% ValidationX2 = X2(indexVX2);
% ValidationTnew = Tnew(indexVTN);
% 
% TestX1 = X1(indexTEX1);
% TestX2 = X2(indexTEX2);
% TestTnew = Tnew(indexTETN);

%% Visualizing the training set

% X1_data = TrainingX1;
% X2_data = TrainingX2;
% Y_data = TrainingTnew;

X1_data = trainingX(1,:).';
X2_data = trainingX(2,:).';
Y_data = trainingY.';

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
xlabel('X1');
ylabel('X2');
zlabel('Tnew');


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

%This section is MSE vs hidden units

X_cell = cell(1, length(X1_data));  % create an empty cell array of size 1x1000

for i = 1:length(X1_data)
    X_cell{i} = [X1_data(i); X2_data(i)];  % store the i-th pair of values in a cell
end

TrainingInput = X_cell;
TrainingTarget = num2cell(trainingY);

% Zie algorim1.m voor voorbeeld hoe dit te doen

alg = 'trainlm';
H = 50; %number of neurons in the hidden layer
delta_epochs = 1000;% Number of epochs to train in each step

net = feedforwardnet(H,alg);
net = configure(net, TrainingInput, TrainingTarget);
net = init(net);
net.trainParam.epochs=delta_epochs;
net=train(net,TrainingInput,TrainingTarget);

postregm(cell2mat(sim(net,trainingP)),trainingY);
postregm(cell2mat(sim(net,validationP)),validationY);

maxHiddenNeurons = 100;
%nets = zeros(length(maxHiddenNeurons));
%simulationTraining = zeros(length(maxHiddenNeurons));
%mseTraining = zeros(length(maxHiddenNeurons));
%simulationValidation = zeros(length(maxHiddenNeurons));
%mseValidation = zeros(length(maxHiddenNeurons));
for i=5:maxHiddenNeurons
    nets{i} = feedforwardnet(i,alg);
    nets{i}.trainParam.epochs=delta_epochs;
    nets{i}=train(nets{i},trainingP,trainingT);
    simulationTraining{i}=sim(nets{i},trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    simulationValidation{i}=sim(nets{i},validationP);
    mseValidation{i} = mean((validationY-cell2mat(simulationValidation{i})).^2);
end


%% Represent here the output surface of the test set 

X1_data = testX(1,:).';
X2_data = testX(2,:).';
Y_data = testY.';

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
xlabel('X1');
ylabel('X2');
zlabel('Tnew');


% Optionally add markers to get an idea of how your data is distributed

hold on;

plot3(X1_data, X2_data, Y_data,'.','MarkerSize',15)



%% look closer between 30 and 50 hidden neurons and let it every time run 10 times

alg = 'trainlm';
H = 50; %number of neurons in the hidden layer
delta_epochs = 1000;% Number of epochs to train in each step

maxHiddenNeurons = 50;
for i=30:maxHiddenNeurons
    nets{i} = feedforwardnet(i,alg);
    nets{i}.trainParam.epochs=delta_epochs;
    nets{i}=train(nets{i},trainingP,trainingT);
    simulationTraining{i}=sim(nets{i},trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    simulationValidation{i}=sim(nets{i},validationP);
    mseValidation{i} = mean((validationY-cell2mat(simulationValidation{i})).^2);
    for j=1:10
        tempnets = feedforwardnet(i,alg);
        tempnets.trainParam.epochs=delta_epochs;
        [tempnets,temptr]=train(tempnets,trainingP,trainingT);
        tempsimulationTraining=sim(tempnets,trainingP);
        tempmseTraining = mean((trainingY-cell2mat(tempsimulationTraining)).^2);
        tempsimulationValidation=sim(tempnets,validationP);
        tempmseValidation = mean((validationY-cell2mat(tempsimulationValidation)).^2);
        if tempmseValidation<mseValidation{i}
            nets{i} = tempnets;
            %tr{i} = temptr;
            simulationTraining{i}=tempsimulationTraining;
            mseTraining{i} = tempmseTraining;
            simulationValidation{i}=tempsimulationValidation;
            mseValidation{i} = tempmseValidation;
        end
    end
end

%% Plot this figure
figure;
hold on;
x=30:50;
plot(x,cell2mat(mseTraining));
plot(x,cell2mat(mseValidation));
set(gca, 'YScale', 'log')
xlabel('Amount of  hidden units');
ylabel('MSE');
legend('Training set','Validation set','Location','northeast');
hold off;

%% Then select the best NN to be made and train this
% we select 43 hidden neurons; vanaf dan begint het te stabiliseren
% Only one hidden layer as it is a good approximator already


hiddenNeurons = 43;
delta_epochs = 1000;
maxFails = 10;

net = feedforwardnet(hiddenNeurons,'trainlm');
net.trainParam.epochs=delta_epochs;
net.trainParam.max_fail = maxFails;

%net.divideFcn = 'divideind';
%net.divideParam.trainInd = trainingP;
%net.divideParam.valInd = validationP;
%net.divideParam.testInd = testP;

net = init(net);
[net, net_Tr ]=train(net,trainingP,trainingT);


simulationTraining2=sim(net,trainingP);
mseTraining2 = mean((trainingY-cell2mat(simulationTraining2)).^2);
simulationValidation2=sim(net,validationP);
mseValidation2 = mean((validationY-cell2mat(simulationValidation2)).^2);

%%
figure;
plotperform(net_Tr);

%% Now train this NN multiple times
hiddenNeurons = 43;
delta_epochs = 1000;

net = feedforwardnet(hiddenNeurons,'trainlm');
net.trainParam.epochs=delta_epochs;
[net,tr]=train(net,trainingP,trainingT);
simulationTraining=sim(net,trainingP);
mseTraining = mean((trainingY-cell2mat(simulationTraining)).^2);
simulationValidation=sim(net,validationP);
mseValidation = mean((validationY-cell2mat(simulationValidation)).^2);
for j=1:10
    tempnets = feedforwardnet(hiddenNeurons,'trainlm');
    tempnets.trainParam.epochs=delta_epochs;
    [tempnets,temptr]=train(tempnets,trainingP,trainingT);
    tempsimulationTraining=sim(tempnets,trainingP);
    tempmseTraining = mean((trainingY-cell2mat(tempsimulationTraining)).^2);
    tempsimulationValidation=sim(tempnets,validationP);
    tempmseValidation = mean((validationY-cell2mat(tempsimulationValidation)).^2);
    if tempmseValidation<mseValidation
        nets = tempnets;
        tr = temptr;
        simulationTraining=tempsimulationTraining;
        mseTraining = tempmseTraining;
        simulationValidation=tempsimulationValidation;
        mseValidation = tempmseValidation;
    end
end

mseValidation %MSE on validation set was: 4.2427e-08

%%
% MSE on the test set:
mseTest = mean((testY-cell2mat(sim(net,testP))).^2) % 3.9360-07

%%
%Plot NN surface of the prediction

x=testX(1,:).';
y=testX(2,:).';
xlin = linspace(min(x),max(x),200);
ylin = linspace(min(y),max(y),200);
[X,Y] = meshgrid(xlin,ylin);
outRows = size(X, 1);
outCols = size(Y, 2);
Z = zeros(outRows, outCols);
for row = 1:outRows
    for col = 1:outCols
        input = [X(row, col); Y(row, col)];
        simulated = sim(net,input);
        Z(row, col) = simulated;
    end
end

figure
%mesh(X,Y,Z,'FaceAlpha',0.7);
mesh(X,Y,Z);
%axis tight; 
xlabel('X1 test set');
ylabel('X2 test set');
zlabel('Tnew predicted');
hold on

%% THIS cell doesnt work!

X1_data = testX(1,:).';
X2_data = testY(1,:).';
Y_data = cell2mat(simulationTestSet).';
%doubleArray = cellfun(@double, simulationTestSet);
%Y_data = doubleArray.';

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
title('The prediction');
xlabel('X1 test set');
ylabel('X2 test set');
zlabel('Tnew predicted');


% Optionally add markers to get an idea of how your data is distributed
hold on;
plot3(X1_data, X2_data, Y_data,'.','MarkerSize',15)

%% Give markers of where the errors are
%Plot error surface


x = testX(1,:).';
y = testX(2,:).';
xlin = linspace(min(x),max(x),40);
ylin = linspace(min(y),max(y),40);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,(testY - cell2mat(sim(net,testP))).^2.');
Z = f(X,Y);
figure;
mesh(X,Y,Z); %interpolated
axis tight; hold on;
contour3(X,Y,Z,100); %Error curves
plot3(trainingX(1,:).',trainingX(2,:).',trainingY.','.','MarkerSize',10); %nonuniform
hold off;

%%
% Create meshgrid
x = testX(1,:).';
y = testX(2,:).';
xlin = linspace(min(x), max(x), 40);
ylin = linspace(min(y), max(y), 40);
[X, Y] = meshgrid(xlin, ylin);

% Calculate squared error
error = (testY - cell2mat(sim(net,testP))).^2.';

% Interpolate error values on the meshgrid
F = scatteredInterpolant(x, y, error);
Z = F(X, Y);

% Create surface plot
figure;
surf(X, Y, Z);
colormap jet;
%colorbar;
grid on;
zlabel('Squared Error');
%title('Squared Error between Test Set and Neural Network Output (Surface Plot)');

%% Trainbfg
% Met noise slechter dan trainlm -> noise zit geen patroon




