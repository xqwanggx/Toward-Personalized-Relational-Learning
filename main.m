
clear,clc
addpath(fullfile('dataset'));
addpath(fullfile('methods'));

load mnist.mat;
X = train_x(1:500,:);
n = size(X,1);
k = max(train_y(1:500))+1;
Y = zeros(n,k);
for i = 1:n
    Y(i,(train_y(i)+1)) = 1;
end

options = [];
options.k = 5;
options.NeighborMode = 'KNN';
options.WeightMode = 'HeatKernel';
A = constructW(X,options);  

alpha = 1;
beta =  1;
gamma = 1;
maxItr = 10;

[Wg,Wl] = PRL(X,Y,A,alpha,beta,gamma,maxItr);









