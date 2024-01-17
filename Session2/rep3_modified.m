%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
clear;
close all;
clc;

T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
n=100;

iterationsToStableRandom = zeros(n, 1);
for i=1:n
    a={rands(3,1)};                         % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 50},{},a);       % simulation of the network  for 50 timesteps
    iterationsToStableRandom(i) = findStableIterations(y);
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle
end


A = {[-1; 1; 1] [-1; -1; -1] [1; 1; -1] [-1; 1; -1]};

iterationsToStableA = zeros(4, 1);
for i=1:length(A)
    a=A(i);                                 % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 50},{},a);       % simulation of the network  for 50 timesteps
    iterationsToStableA(i) = findStableIterations(y);
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,50),record(2,50),record(3,50),'gO');  % plot the final point with a green circle
end

grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
%title('Time evolution in the phase space of 3d Hopfield model');


avgIterationsToStableRandom = mean(iterationsToStableRandom);
avgIterationsToStableA= mean(iterationsToStableA);
fprintf('Average iterations to reach a stable state for the random points: %.2f\n', avgIterationsToStableRandom);
fprintf('Average iterations to reach a stable state for the fixed points in cell array A: %.2f\n', avgIterationsToStableA);


function iterations = findStableIterations(sequence)
    iterations = 0;
    numIterations = size(sequence, 2);
    
    for i = 2:numIterations
        if isequal(sequence(:, i), sequence(:, i-1))
            iterations = i;
            break;
        end
    end
end



