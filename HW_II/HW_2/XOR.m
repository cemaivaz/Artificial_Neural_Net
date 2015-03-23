close all;
clear all;
clc



format long

fprintf('Model is being trained..\n\n');

%Iteration number
iterNo = 30;


%Hidden unit numbers are shown below
Hidd = [2; 4; 8];

%Training data
dataTr = textread('training.txt', '%s');

%The x values thereof
xt = dataTr(1:2:length(dataTr) - 1);

%The y values thereof
rt = dataTr(2:2:length(dataTr));

%Data are shuffled
randOrd = randperm(length(xt));

tmpXt = [];
tmprt = [];
for i = 1:length(xt)
    tmpXt = [tmpXt; str2double(xt(i))];
    tmprt = [tmprt; str2double(rt(i))];
    
end



xt = tmpXt;
rt = tmprt;

xt = xt(randOrd);

rt = rt(randOrd);


x = [0 0 1 1; 0 1 0 1];



t = [0 1 1 0];


x = [1 1 2 4 3];

t = [0.11 0.14 0.55 0.98 0.77];

% x = xt';
% t = rt';

[ni N] = size(x)

[no N] = size(t)

nh = 2

% wih = .1*ones(nh,ni+1);

% who = .1*ones(no,nh+1);

wih = 0.02*randn(nh,ni+1) - 0.01;

who = 0.02*randn(no,nh+1) - 0.01;

 c = 0;
 while(c < 9000)
     c = c+1;
% %for i = 1:length(x(1,:))

     for i = 1:N    
        for j = 1:nh
            netj(j) = wih(j,1:end-1)*x(:,i)+wih(j,end);
% %outj(j) = 1./(1+exp(-netj(j)));

            outj(j) = tansig(netj(j));
        end
        % hidden to output layer
        for k = 1:no             
           netk(k) = who(k,1:end-1)*outj' + who(k,end);
           outk(k) = 1./(1+exp(-netk(k))); 
           delk(k) = outk(k)*(1-outk(k))*(t(k,i)-outk(k)); 
        end
         % back propagation 
        for j = 1:nh
            s=0; 
            for k = 1:no 
               s = s + who(k,j)*delk(k); 
            end
            delj(j) = outj(j)*(1-outj(j))*s; 
% %s=0;

        end 
        for k = 1:no
            for l = 1:nh
                who(k,l) = who(k,l)+.5*delk(k)*outj(l);
            end
            who(k,l+1) = who(k,l+1)+1*delk(k)*1;
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

e = sum((t-y).^ 2);