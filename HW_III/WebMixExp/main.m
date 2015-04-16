clear all
close all

%toy data 
[Target, X,PHI,W] = TwoExpertToyData(  );
    
    K=2;
%Initialization for Expert%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 


MaxIterations=100; 

NumberExperts=2


RegulationTermDesign=0.01;
RegulationTermCovariance=0.01;


[Prior, Mu, Sigma,West,Var,Lickyhood] =TrainMixtureofExpertsParametric(Target,PHI,X,NumberExperts,RegulationTermDesign,RegulationTermCovariance);

for n=1:3
   
    [Prior0, Mu0, Sigma0,West0,Var0,Lickyhood0] =TrainMixtureofExpertsParametric(Target,PHI,X,NumberExperts,RegulationTermDesign,RegulationTermCovariance);
    
    if Lickyhood0(end)>Lickyhood(end)
    Prior=Prior0
    Mu=Mu0
    Sigma=Sigma0
    West=West0;
    Var=Var0;
    
     Lickyhood=Lickyhood0;
    
    end
    
    
end

[ TotalOutput,ExpertOutput,GateProbability] = OutputMixtureofExpertsParametric(Prior, Mu, Sigma,West,X,PHI)
figure
plot(Lickyhood)

title('Log Likelihood of every iteration of EM algorithm' )
xlabel('iteration ')


%%


figure
subplot(211)
 plot(X(:,2),TotalOutput,'k','LineWidth',3)
 hold on 
 plot(X(:,2),ExpertOutput,'--','LineWidth',3)
  plot(X(:,2),Target,'ro','LineWidth',2)
  legend('Total Output','Output Expert 1','Output Expert 2','Target')
  xlabel('x ')
 


  hold off
  subplot(212)
   plot(X(:,2),GateProbability)
 legend('Gating function 1','Gating function 2')
    
    