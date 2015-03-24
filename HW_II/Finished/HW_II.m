
%Cem Rifki Aydin    2013800054
%CmpE545    Hw 2
%13.03.2015


close all;
clear all;
clc

format long


NH = [2; 4; 8];


thresh = 1500;

fprintf('Model is being trained..\n\n');


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

x = xt(randOrd)';

t = rt(randOrd)';


%VALIDATION DATA
%Validation data
dataVal = textread('validation.txt', '%s');

xtVal = dataVal(1:2:length(dataVal) - 1);

rtVal = dataVal(2:2:length(dataVal));

randOrd = randperm(length(xtVal));


tmpXtVal = [];
tmprtVal = [];
for i = 1:length(xtVal)
    tmpXtVal = [tmpXtVal; str2double(xtVal(i))];
    tmprtVal = [tmprtVal; str2double(rtVal(i))];
    
end

xtVal = tmpXtVal;
rtVal = tmprtVal;



xVal = xtVal(randOrd)';
tVal = rtVal(randOrd)';

[ni N] = size(x)

[no N] = size(t)



for hiddNo = 1:length(NH)
    nh = NH(hiddNo);
    
    wih = 0.02*randn(nh,ni+1) - 0.01;
    
    who = 0.02*randn(no,nh+1) - 0.01;
    
    n = 0.1; %learning parameter

    
    error = [];
    
    errorVal = [];
    
    for c = 0:1:thresh
        for i = 1:N
            for j = 1:nh
                
                netj(j) = wih(j,1:end-1)*x(:,i)+wih(j,end);
                
                outj(j) = 1./(1+exp(-netj(j)));%logsig(netj(j));
            end
            % hidden to output layer
            for k = 1:no
                
                outk(k) = who(k,1:end-1)*outj' + who(k,end);
                delk(k, :) = n * (t(k, i) - outk(k));
            end
            % back propagation
            for j = 1:nh

                delj(j) =  n * (t(i) - outk(k)) * who(j) * outj(j) * (1 - outj(j));
            end

            for k = 1:no
                for l = 1:nh
                    who(k, l) = who(k, l) + delk(k) * outj(l);
                end

                who(k, l + 1) = who(k, l + 1) + delk(k) * 1;
                
            end
            for j = 1:nh

                
                for ii = 1:ni
                    wih(j,ii) = wih(j,ii)+delj(j) .* x(ii,i);
                end
                
                wih(j,ii+1) = wih(j,ii+1)+1*delj(j);
                
            end
        end
        
        h = logsig(wih * [x; ones(1, N)]);
        
        y = who * [h; ones(1, N)];
        
        
        err = t-y;
        
        sum_ = sum(err.^ 2);
        
        error = [error sum_];
        
        
        %Validation
        hVal = logsig(wih * [xVal; ones(1, N)]);
        
        yVal = who * [hVal; ones(1, N)];
        
        
        err = tVal-yVal;
        
        sum_ = sum(err.^ 2);
        
        errorVal = [errorVal sum_];
    end
    
    
    e = t-y;
    
    sum(e.^ 2)
    
    figure();
    
    x_ = linspace(min(xt), max(xt), 100);
    
    len = length(x_);
    h_ = logsig(wih * [x_; ones(1, len)]);
    

    y_ = who * [h_; ones(1,len)]
    plot(x_, y_, '-');
    hold on;
    plot(xt, rt, '+');
    hold on;
    for hLine = 1:size(wih, 1)
        plot(x_, wih(hLine, :) * [x_; ones(1, len)]);
        hold on;
    end
    
    hold off;
    
    figure();

    
    xAxis = 1:length(error);
    plot(xAxis, error, '-')
    
    hold on;
    plot(xAxis, errorVal, '-r')
        
    hold off;
end
