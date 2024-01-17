%%%%%%%%%%%
% rep2.m
% A script which generates n random initial points 
%and visualises results of simulation of a 2d Hopfield network 'net'
%%%%%%%%%%

clear;
clc;
close all;

T = [1 1; -1 -1; 1 -1]';
net = newhop(T);
n=150;

%yAllRandom = cell(50, n); %Note the 50 steps!

iterationsToStableRandom = zeros(n, 1);
for i=1:n % random points
    a={rands(2,1)};                     % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);   % simulation of the network for 50 timesteps 
    iterationsToStableRandom(i) = findStableIterations(y);
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end

n=10;
iterationsToStableX = zeros(n, 1);
for i=1:n % Parallel to X
    temp=2/n * i -1;
    a={[temp;0]};                       % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);   % simulation of the network for 50 timesteps 
    iterationsToStableX(i) = findStableIterations(y);
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'b'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end


n=10;
iterationsToStableY = zeros(n, 1);
for i=1:n %Parallel to Y
    temp=2/n * i -1;
    a={[0;temp]};                     % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);   % simulation of the network for 50 timesteps   
    iterationsToStableY(i) = findStableIterations(y);
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'b'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end

legend('initial state','time evolution','attractor','Location', 'northeast');
%title('Time evolution in the phase space of 2d Hopfield model');

avgIterationsToStableRandom = mean(iterationsToStableRandom);
avgIterationsToStableX = mean(iterationsToStableX);
avgIterationsToStableY = mean(iterationsToStableY);

fprintf('Average iterations to reach a stable state for the random points: %.2f\n', avgIterationsToStableRandom);
fprintf('Average iterations to reach a stable state for the points parallel to X axis: %.2f\n', avgIterationsToStableX);
fprintf('Average iterations to reach a stable state for the points parallel to Y axis: %.2f\n', avgIterationsToStableY);

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



