clear
clc
close all

%% On Gaussian random numbers

%Generate the dataset
dataset = randn(50,500);

mean_vector = mean(dataset,2);
zero_dataset = dataset - mean_vector;

covariance = cov(zero_dataset');
    
%Compute variance for each dimension
values = eig(covariance);
sum_total_values = sum(values);

errors = zeros(1,49);
var_subs = zeros(1, 49);
for q=1:49 % Q = 
    
    %Compute the reduced eigenvalues and eigenvactors of the covariance matrix
    [E, reduced_values] = eigs(covariance, q);
    sum_reduced_values = sum(diag(reduced_values));
    
    %Compute the percentage of variance captured by subspace
    var_sub = sum_reduced_values/sum_total_values;
    var_subs(1,q) = var_sub;
    
    %Compute transpose of reduced eigenvectors
    E_T = E';
    
    %Reduce the dataset by multipling with E_T
    reduced_dataset = E_T*dataset;
    
    %Generate new dataset from reduced dataset
    new_dataset_zerod = E*reduced_dataset;
    new_dataset = new_dataset_zerod + mean_vector;
    
    %Estimate the error
    RMSE = (sqrt(mean(mean((dataset-new_dataset).^2))));
    
    errors(1,q) = RMSE;
end

%Plotting RMSE for different q-values

figure;
hold on;
plot(errors);
%plot(var_subs);
ylabel('RMSD')
xlabel('Dimension of projection')
xlim([1 51]);
ylim([0 1.1]);
%legend({'RMSE','Variance (%)'},'Location','north')
hold off;


%% On choles_all dataset

load choles_all
mean_vector = mean(p,2);
p_zero_mean = p - mean_vector;
covariance_matrix = cov(p_zero_mean');
[V,D] = eigs(covariance_matrix, 21);
eigen_values = diag(D);
RMSE = zeros(20,1);

for i=1:20
    E = V(:, 1:i);
    z = E'*p_zero_mean;
    p_hat = E * z + mean_vector;
    RMSE(i) = sqrt(mean(mean((p-p_hat).^2)));
end

figure;
plot(RMSE)
xlabel('Dimension of projection')
ylabel('RMSD')


