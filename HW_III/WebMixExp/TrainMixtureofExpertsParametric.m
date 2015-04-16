

function [Prior, Mu, Sigma,West,Var,Lickyhood] =TrainMixtureofExpertsParametric(Target,PHI,X,NumberExperts,RegulationTermDesign,RegulationTermCovariance,MaxIterations)

%This code implements an alternative model for mixtures of experts using a Gaussian for the gating function.
%There form is used so that the maximization with respect to the parameters of the gating network can be handled analytically.
%Thus, a single-loop EM can be used, and no learning stepsize is required to guarantee convergence. The method is based on 
%: An Alternative Model for Mixtures of Experts by Lei Xu, Michael!. Jordan, Geoffrey E. Hinton
% The code also uses Sylvain Calinon Expectation-Maximization (EM)
% algorithm On Learning, Representing and Generalizing a Task in a Humanoid Robot","S. Calinon and F. Guenter and A. Billard",
 
% Inputs -----------------------------------------------------------------
%Target:N x 1 array representing N data points of targets that are ment to be Estimated.
% PHI: N x M array representing N data points of M dimensions that consist of some nonlinear
%transformation of the array X  
%X:N x D array representing N datapoints of D dimensions 
%NumberExperts: Integer indicating number of experts  
%RegulationTermDesign: Value for diagonal regularisation term for the
%design matrix
%RegulationTermCovariance:Regularization term for design matrix   
%MaxIterations: maxmum number of maximum number of iterations: Default 100  
% Outputs ----------------------------------------------------------------
%Gate Parameters 
% Priors:  1 x NumberExperts array representing the prior probabilities of the K GMM 
% components.
% Mu:      D x NumberExperts array representing the centers of the K GMM components.
% Sigma:   D x D x NumberExperts array representing the covariance matrices of the 
%              K GMM components.
%Expert Parameters 
%West: MxnumberExperts array Representing the number of parameters, each column represents the parameters for a 
%different expert, analogues to the parameters in linear regression   
%Var: variance of each expert 

% Author:Joseph Santarcangelo  29/10/2014
%Vertion one 
%Also see:InitializationOfExpert,gaussPDF,


if (nargin<=7)
    
      MaxIterations=100;  
end
  
if (nargin<=6)
    
      MaxIterations=100;  
RegulationTermDesign=0.01;

end  
if (nargin<=6)
    
      MaxIterations=100;  
RegulationTermDesign=0.01;

end  
NumberDataPoint=length(X(:,1));

DimensionsofData=length(X(1,:));

%% Criterion to stop the EM iterative update
loglik_threshold = 1e-10;

%Inshilzation Gate Parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  

[ Prior, Mu, Sigma,West,Var,InitializeLikelihood] = InitializationOfExpert( X,PHI,NumberExperts,Target);


    

%Regularization term for design matrix for expert 
IregDesign=RegulationTermDesign*eye(length(PHI(1,:)));

%Regularization term for design for covariance  matrix for gating 
IregCov=RegulationTermCovariance*eye(length(X(1,:)));



loglik_old = InitializeLikelihood;
nStep=1;
%Initialization for E-step
%Lick hood of  X given of expert each row represents the probability of a data point 


%Probabilities for each gate for every sample xn, each row represents a
%sample, each column represents a sample, for colunm k 
%:[p(x1|z=k),p(x2|z=k),..,p(xN|z=k)]
Pxz=zeros(NumberDataPoint,NumberExperts);

Pyzx=zeros(NumberDataPoint,NumberExperts);
 % Probability of each expert given the parameters and target 
Pyx=zeros(NumberDataPoint,NumberExperts);
ProductLikelihoodzeros=zeros(NumberDataPoint,1);
%dummy vector 

Lickyhood=[];

Xtemp=zeros(size(X));

Inshilize_loglike_old =InitializeLikelihood;
Inshilize_loglike=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
while 1


%E-step
for expert=1:NumberExperts
    
 %Compute probability p(x|z)
 Pxz(:,expert) = gaussPDF(X, Mu(:,expert), Sigma(:,:,expert));
 
 %Compute probability p(y|yhat,x) expert paramter 
 Pyzx(:,expert)= ConditionalPDFExpert ( Target,PHI*West(:,expert),Var(expert));
end

 %Compute posterior probability p(z|x,y)or
Pyx=Pyzx.*Pxz.*repmat(Prior,NumberDataPoint,1);
 % h(y|x)=p(z|x,y)
 Pyx=Pyx./((repmat(sum(Pyx')',1,NumberExperts))+realmin);

E = sum(Pyx);

 for i=1:NumberExperts
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     %Update the priors
    Priors(i) = E(i) / NumberDataPoint;
    %Update the centers
    Mu(:,i) = Pyx(:,i)'*X /( E(i)+realmin);      
    Id=diag(Pyx(:,i)); 
    
    West(:,i)=(((PHI'*Id*PHI+IregDesign))^-1)*PHI'*Id*Target;
    Var(i)=Pyzx(:,i)'*((Target-PHI*West(:,i)).^2)/( E(i)+realmin);
    
    %Update the covariance matrices
    Xtemp = X - repmat(Mu(:,i)',NumberDataPoint,1);
    Sigma(:,:,i) = repmat(Pyx(:,i)',DimensionsofData,1).* Xtemp'* Xtemp / (E(i)+realmin);
    % Add a tiny variance to avoid numerical instability
    Sigma(:,:,i) = Sigma(:,:,i) +IregCov;
    %
 end

 
 for expert=1:NumberExperts
    
 %Compute probability p(x|z)
 Pxz(:,expert) = gaussPDF(X, Mu(:,expert), Sigma(:,:,expert));
 %Compute probability p(y|yhat,x)
 Pyzx(:,expert)= ConditionalPDFExpert ( Target,PHI*West(:,expert),Var(expert));

end

  %Compute the log likelihood (the matrix operation takes care of the sum )
   ProductLikelihood =Pyzx.*Pxz*Prior' ;
  
  ProductLikelihood(find(ProductLikelihood<realmin)) = realmin;
 
  loglik = mean(log(ProductLikelihood));
  Lickyhood( nStep)=loglik;

  %Stop the process depending on the increase of the log likelihood 
 if abs((loglik/loglik_old)-1) < loglik_threshold || nStep==MaxIterations 
    break;
  end
  loglik_old = loglik;
  nStep = nStep+1;
end

end
function [ Prior, Mu, Sigma,West,Var,loglik] = InitializationOfExpert( X,PHI,NumberExperts,Target)
%This function inshilzation expert parameters first by Initializes the mixture of experts
%by using k-means to means to initialize the gate , then using the hard labels of sample to train expert
%Input
%X: feature space for expert
%PHI:feature space for regression   
%Target:target for expert 
%NumberExperts: Number eof experts
%Outputs
%Prior:Parameter for gaining function, analogous to posterior in Bayesian likelihood  
%Mu: mean of liclyhood 
%Sigma: Covariance matrix for each gate
%West: Linear repression parameter of each expert   
%Var: 


%error of  expert
Var=rand(1,NumberExperts);
%Parameters for expert
West=zeros(length(PHI(1,:)),NumberExperts);

NumberDataPoint=length(X(:,1));

%k-means for gating function
[Prior, Mu, Sigma,ExpertInitialization] = EM_init_kmeans(X', NumberExperts);
%Matrix used for hard thresholds
IndictorMatrix=zeros(NumberDataPoint,NumberDataPoint);



for expert=1:NumberExperts
    
    IndictorMatrix(:,:)=diag(ExpertInitialization==expert);
    West(:,expert)=(((PHI'*IndictorMatrix*PHI))^-1)*PHI'*IndictorMatrix*Target;
    
    Var(1,expert)=sum((Target(ExpertInitialization==expert)-PHI(ExpertInitialization==expert,:)*West(:,expert)).^2)/sum(ExpertInitialization==expert);
    
end

for expert=1:NumberExperts
    
    %Compute probability p(x|z)
    Pxz(:,expert) = gaussPDF(X, Mu(:,expert), Sigma(:,:,expert));
    
    %Compute probability p(y|yhat,x)
    Pyzx(:,expert)= ConditionalPDFExpert ( Target,PHI*West(:,expert),Var(expert));
    
end

%Compute the log likelihood
ProductLikelihood =Pyzx.*Pxz*Prior'; 
  
loglik = mean(log(ProductLikelihood));
end
function [Priors, Mu, Sigma,Data_id] = EM_init_kmeans(Data, nbStates)
%
% This function initializes the parameters of a Gaussian Mixture Model 
% (GMM) by using k-means clustering algorithm.
%
% Author:	Sylvain Calinon, 2009
%			http://programming-by-demonstration.org
%
% Inputs -----------------------------------------------------------------
%   o Data:     D x N array representing N datapoints of D dimensions.
%   o nbStates: Number K of GMM components.
% Outputs ----------------------------------------------------------------
%   o Priors:   1 x K array representing the prior probabilities of the
%               K GMM components.
%   o Mu:       D x K array representing the centers of the K GMM components.
%   o Sigma:    D x D x K array representing the covariance matrices of the 
%               K GMM components.
% Comments ---------------------------------------------------------------
%   o This function uses the 'kmeans' function from the MATLAB Statistics 
%     toolbox. If you are using a version of the 'netlab' toolbox that also
%     uses a function named 'kmeans', please rename the netlab function to
%     'kmeans_netlab.m' to avoid conflicts. 

[nbVar, nbData] = size(Data);

%Use of the 'kmeans' function from the MATLAB Statistics toolbox
[Data_id, Centers] = kmeans(Data', nbStates); 
Mu = Centers';
for i=1:nbStates
  idtmp = find(Data_id==i);
  Priors(i) = length(idtmp);
  Sigma(:,:,i) = cov([Data(:,idtmp) Data(:,idtmp)]');
  %Add a tiny variance to avoid numerical instability
  Sigma(:,:,i) = Sigma(:,:,i) + 1E-5.*diag(ones(nbVar,1));
end
Priors = Priors ./ sum(Priors);
end


function prob = gaussPDF(Data, Mu, Sigma)
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

function [ Out ] = ConditionalPDFExpert ( RealValue,EstimatedValue,Var)


%This function produces  conditional mixture for each expert for mixture of experts  
%Input
%RealValue:Column vector of Target value 
%EstimatedValu:Column vector of Estimated value 
%Var:Variance of data 
%Output
%Out:Probability of each expert  

Out = (RealValue-EstimatedValue).^2;

Out=Out/(Var); 
Out = exp(-0.5*Out) / (sqrt(2*pi*Var)+realmin);


end




