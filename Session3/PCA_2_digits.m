close all;
clear;
clc;

load('threes.mat','-ascii');
image_no=1;
colormap('gray');
imagesc(reshape(threes(image_no,:),16,16),[0,1]);

%Display the average "3"
average = mean(threes,1);
colormap('gray');

%% Loading data
load threes -ascii

% Visualising image
colormap('gray');
imagesc(reshape(threes(1,:),16,16),[0,1]);

% Calculating and visualising the mean '3'
mean3 = mean(threes);
imagesc(reshape(mean3(1,:),16,16),[0,1]);

%% plotting eigenvalues

%Changing variable name for readability
data = threes';

%Calculate the mean of each column
mean_vector = mean(data,1);

%Zero mean the dataset by subtracting the mean
zero_data = data - mean_vector;

% Computing the covariance matrix
covariance = cov(zero_data');

% Computing eigenvectors and eigenvalues of covariance matrix
[vectors,values] = eig(covariance);
eigenvalues = diag(values);
sum_eigenvalues = sum(eigenvalues);

% Plotting eigenvalues
figure;
plot(eigenvalues);
xlabel("Dimension");
ylabel("Eigenvalue");

%% Compress in one, two, three, and four principal components

maxQ = size(threes,2);
zeroMeanD = threes - mean(threes); %Column-wise mean

for i=1:maxQ
    covarianceMatrix = cov(zeroMeanD);
    [eigenvecs,eigenvals] = eigs(covarianceMatrix,i); %Eigenvectors are column vectors in the matrix
    eigenvals = diag(eigenvals);

    transformedDataset = eigenvecs.' * zeroMeanD.';

    reconstructedZeroMeanDataset{i} = eigenvecs * transformedDataset;

    reconstructedZeroMeanDataset{i} = reconstructedZeroMeanDataset{i}.';

    reconstructionError(i) = sqrt(mean(mean((zeroMeanD-reconstructedZeroMeanDataset{i}).^2))); %Root mean square difference
    %i
    
    reconstructedDataset{i} = reconstructedZeroMeanDataset{i} + average;
end

image_no = 10;
figure;
s = subplot(2,2,1);
colormap('gray');
subplot(2,2,1);
imagesc(reshape(reconstructedDataset{1}(image_no,:),16,16),[0,1]);
title('1 eigenvalue');
axis image;
subplot(2,2,2);
imagesc(reshape(reconstructedDataset{2}(image_no,:),16,16),[0,1]);
title('2 eigenvalues');
axis image;
subplot(2,2,3);
imagesc(reshape(reconstructedDataset{3}(image_no,:),16,16),[0,1]);
title('3 eigenvalues');
axis image;
subplot(2,2,4);
imagesc(reshape(reconstructedDataset{4}(image_no,:),16,16),[0,1]);
title('4 eigenvalues');
axis image;

% TODO hierboven ook nog de reconstruction error geven telkens!

%% compress to Q principal components, plot reconstruction error as a function of q

%Setting size of projected dimension
q = 256;

%Initialise errors and eigenvalue-percentages arrays
errors = zeros(1,q);
var_subs = zeros(1,q);

for i=1:q
    %Compute the reduced eigenvalues and eigenvactors of the covariance matrix
    [E, reduced_values] = eigs(covariance, i);
    sum_reduced_values = sum(diag(reduced_values));
    
    %Compute the percentage of variance captured by subspace
    var_sub = sum_reduced_values/sum_eigenvalues;
    var_subs(1,i) = var_sub;
    
    %Compute transpose of reduced eigenvectors
    E_T = E';
    
    %Reduce the dataset by multipling with E_T
    reduced_data = E_T*data;
    
    %Generate reconstructed dataset from reduced dataset
    reconstructed_data_0 = E*reduced_data;
    reconstructed_data = reconstructed_data_0 + mean_vector;
    
    %Estimate the error
    RMSD = sqrt(mean(mean((data-reconstructed_data).^2)));
    errors(1,i) = RMSD;
end

max(var_subs);
min(errors);

error_end = errors(1,end-1);

%Plotting RMSD for different q-values
plot(errors);
hold on;
xlabel('Dimension of projection')
ylabel('RMSD');
title("")
xlim([1 q]);
ylim([0 0.4]);
hold off;

%% Cumulative sum of principal components

eigen_values = eigenvalues';
eigen_vector_new = zeros(size((eigen_values)));

for i=1:256
    %eigen_vector_new(end-i+1) = eigen_values(257-i);
    eigen_vector_new(i) = eigen_values(257-i);
end

cumulative_eigen_value= cumsum(eigen_vector_new); 

figure
plot(cumulative_eigen_value)
title('Cumulative Eigenvalues')
xlabel('Dimension of projection')
grid on;




