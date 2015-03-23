close all;
clear all;
clc

x = [0 0 1 1; 0 1 0 1];



t = [0 1 1 0];


x = [1 1 2 4];

t = [0.11 0.14 0.55 0.98];

[ni N] = size(x)

[no N] = size(t)

nh = 2

% wih = .1*ones(nh,ni+1);

% who = .1*ones(no,nh+1);

wih = 0.02*randn(nh,ni+1) - 0.01;

who = 0.02*randn(no,nh+1) - 0.01;

n = 0.1; %learning parameter
 c = 0;
 while(c < 3000)
     c = c+1;
% %for i = 1:length(x(1,:))

     for i = 1:N    
        for j = 1:nh
            netj(j) = wih(j,1:end-1)*x(:,i)+wih(j,end);
% %outj(j) = 1./(1+exp(-netj(j)));

            outj(j) = logsig(netj(j));
        end
        % hidden to output layer
        for k = 1:no            
           netk(k) = who(k,1:end-1)*outj' + who(k,end);
           outk(k) = netk(k);%1./(1+exp(-netk(k))); 
           delk(k, :) = n * (t(i) - outk(k)) * outj;%outk(k)*(1-outk(k))*(t(k,i)-outk(k)); 
        end
         % back propagation 
        for j = 1:nh
            
            s=0; 
            for k = 1:no 
               s = s + who(k,j)*delk(k); 
            end
            delj(j) = outj(j)*(1-outj(j))*s; 
% %s=0;
            delj(j) = n * (t(i) - outk(k)) * delk(j) * outj(j) * (1 - outj(j)) * x(:, i);
        end 
        for k = 1:no
            for l = 1:nh
                %who(k,l) = who(k,l)+.5*delk(k)*outj(l);
                who(k, l) = who(k, l) + delk(k);
            end
            %who(k,l+1) = who(k,l+1)+1*delk(k)*1;
            
            who(k, l + 1) = who(k, l + 1) + delk(k);
            
        end  
        for j = 1:nh
            for ii = 1:ni
                wih(j,ii) = wih(j,ii)+.5*delj(j)*x(ii,i);
            end
            wih(j,ii+1) = wih(j,ii+1)+1*delj(j)*1;
         end    
      end
   end
h = tansig(wih*[x;ones(1,N)])

y = logsig(who*[h;ones(1,N)])

e = t-round(y)