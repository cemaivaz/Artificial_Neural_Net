function [Target, X,PHI,W] = TwoExpertToyData(  )
%%This Function generates some toy data with  two linear experts  

%Output
% PHI: N x M array representing N data points of M dimensions that consist of some nonlinear
%transformation of the array X is this case a second order polinomal  
%X:N x D array representing N datapoints of D dimensions 
%W: (M)x(number Experts) array Representing the number of parameters, each column represents the parameters for a 
%different expert, analogues to the parameters in linear regression   
%Target: target data
z=[-5:0.1:5-0.1];
NumberSamples=length(z);
%Generate uniform data for form -5 to 5
X=ones(2,NumberSamples);
X(2,:)=z;
X=X';
% number of samples
M=length(X(:,1));
N=length(X(1,:));
%Parameters to make data linearly separable

%Generate lables  sample data
Lables=ones(M,1);

Lables(z<0)=2;

Target=ones(M,1);


    degree=2 

PHI=ones(M,3);

PHI(:,2)=X(:,2);
PHI(:,3)=X(:,2).^2;


    
    %make some data with nice parmeters
    W=zeros(3,2)
    W(:,1)=[1 2 3]';
    W(:,2)=-[30 1 1]';
    % Generate some target data
    %noise standard deviation
    Nsd=0.01;
    for n=1:2
        Target(Lables==n)=PHI(Lables==n,:)*W(:,n)+Nsd*randn(sum(Lables==n),1);
    end




end

