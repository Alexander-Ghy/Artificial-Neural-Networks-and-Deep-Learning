%% Set up
clear
clc
close all

%% Section 1
% Create a perceptron and train it with examples (see demos). Visualize results. 
% Can you always train it perfectly?

X = [ -0.5 -0.5 +0.3 -0.1 -40; ...
      -0.5 +0.5 -0.5 +1.0 50];
T = [1 1 0 0 1];
plotpv(X,T);

net = perceptron('hardlim','learnpn'); %hardlim makes output values only 0 or 1 and input vaues can have anything.
net = configure(net,X,T);

hold on
linehandle = plotpc(net.IW{1},net.b{1});

E = 1;
while (sse(E))
   [net,Y,E] = adapt(net,X,T);
   linehandle = plotpc(net.IW{1},net.b{1},linehandle);
   drawnow;
end

hold on;
plotpv(X,T);
plotpc(net.IW{1},net.b{1});
hold off;

