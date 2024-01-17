%%
clc;
clear;
close all;

%%

T = [1 1; -1 -1; 1 -1]';

% Spurious attracter, zie theorie hopfield. het maakt zelf een extra
% attractor als het anders geen stabiel punt kan vinden.

a1 = {[0.25; 0.25]};
a2 = {[-0.25; -0.25]};
a3 = {[-0.25; 0.25]};
a4 = {[0.25; -0.25]};
a5 = {[0 ; 0 ]};
a6 = {[0; -0.5]};
a7 = {[0.5; -0.75]};

net = newhop(T);

A = [a1 a2 a3 a4 a5 a6 a7];

for i=1:length(A)
    a=A(i);                             % generate an initial point 
    [y,Pf,Af] = sim(net,{1 50},{},a);   % simulation of the network for 50 timesteps  
    record=[cell2mat(a) cell2mat(y)];   % formatting results  
    start=cell2mat(a);                  % formatting results 
    plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:),'r'); % plot evolution
    hold on;
    plot(record(1,50),record(2,50),'gO');  % plot the final point with a green circle
end
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 2d Hopfield model');


