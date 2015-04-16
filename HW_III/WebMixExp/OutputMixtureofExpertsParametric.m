function [ TotalOutput,ExpertOutput,GateProbability] = OutputMixtureofExpertsParametric(Prior, Mu, Sigma,West,X,PHI)
%This function provides the output and output components of the mixture of experts code for linear regression using a Gaussian mixture model as the gating function 

% Inputs -----------------------------------------------------------------
% PHI: N x M array representing N data points of M dimensions that consist of some nonlinear
%transformation of the array X  
%X:N x D array representing N datapoints of D dimensions   
%Gate Parameters 
% Priors:  1 x NumberExperts array representing the prior probabilities of the K GMM 
% components.
% Mu:      D x NumberExperts array representing the centers of the K GMM components.
% Sigma:   D x D x NumberExperts array representing the covariance matrices of the 
%Expert Parameters 
%West: MxnumberExperts array Representing the number of parameters, each column represents the parameters for a 
%different expert, analogues to the parameters in linear regression   

%Ouput--------------------------------------------------------------------------------------------------------
 %TotalOutput:Column vector representing the output of the network, where each row represents the output of a different observation 
 %ExpertOutput:(Number of Observations)x(Number of sample) matrix where each column represents the output of each expert and the row corresponds to an observation   
 %GateProbability:(Number of Observations)x(Number of sample) matrix where each column represents expert and the row corresponds to the probability observation   
 %Author:Joseph Santarcangelo, 2014 
 %Contributions :Sylvain Calinon, 2
 
 M=length(X(:,1));
NumberExperts=length(Prior);
ExpertOutput=zeros(M,NumberExperts);
GateProbability=zeros(M,NumberExperts); 
TotalOutput=zeros(M,1);

Normalization=zeros(M,1);
%Output of each expert 
ExpertOutput=PHI*West;

for n=1:NumberExperts
    %Probability of each gate for every sample given the Gaussian model  
    GateProbability(:,n)=gaussPDFOut(X, Mu(:,n), Sigma(:,:,n))*Prior(n);
    %Total output of each expert waited by the probability of that observations 
    TotalOutput=TotalOutput+ExpertOutput(:,n).*GateProbability(:,n);
    %Normalization values 
    Normalization= Normalization+GateProbability(:,n);
end

%Gate  normalized gate values for each expert and each samples 
GateProbability=GateProbability./repmat(Normalization,1,NumberExperts);
%Normalize output 
TotalOutput=TotalOutput./Normalization;
end

function prob = gaussPDFOut(Data, Mu, Sigma)
%
% This function computes the Probability Density Function (PDF) of a
% multivariate Gaussian represented by means and covariance matrix.
%
% Author:	Sylvain Calinon, 2009
%			http://programming-by-demonstration.org
%
% Inputs -----------------------------------------------------------------
%   o Data:  N x D array representing N datapoints of D dimensions.
%   o Mu:    D x 1 array representing the centers of the K GMM components.
%   o Sigma: D x D x K array representing the covariance matrices of the 
%            K GMM components.
% Outputs ----------------------------------------------------------------
%   o prob:  1 x N array representing the probabilities for the 
%            N datapoints.     

[nbData,nbVar] = size(Data);

Data = Data - repmat(Mu',nbData,1);
prob = sum((Data*inv(Sigma)).*Data, 2);
prob = exp(-0.5*prob) / sqrt((2*pi)^nbVar * (abs(det(Sigma))+realmin));
end

