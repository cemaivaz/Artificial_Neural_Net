
%Cem Rifki Aydin    2013800054
%CmpE545    Hw 2
%20.03.2015


close all;
clear all;
clc

format long

%Hidden unit numbers are shown below
NH = [2; 4; 8];

%The number of epochs is shown below
thresh = 800;

fprintf('Model is being trained..\n\n');



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


%Validation data are being read

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

[inpN N] = size(x);

[nOut N] = size(t);


errorAll = zeros(2, length(NH));
for hiddNo = 1:length(NH)
    noH = NH(hiddNo);
    
     %Whj and Vih matrices are filled with random values
    whj = 0.02*randn(noH,inpN+1) - 0.01;
    
    vih = 0.02*randn(nOut,noH+1) - 0.01;
    
    n = 0.1; %learning parameter

    
    error = []; %Error for training data
    
    errorVal = []; % Error for validation data
    
    for c = 0:1:thresh
        for i = 1:N
            for j = 1:noH
                
                befSigm(j) = whj(j,1:end-1) * x (:,i) + whj(j,end);
                
                %Sigmoid value being calculated below
                Zh(j) = 1./(1+exp(-befSigm(j)));
            end
            %The regression output value being calculated below
            for k = 1:nOut
                
                output(k) = vih(k,1:end-1)*Zh' + vih(k,end);
                %DeltaVih values being calculated below
                delVih(k, :) = n * (t(k, i) - output(k));
            end
            %DeltaWhj values for backpropagation being calculated below
            for j = 1:noH

                delWhj(j) =  n * (t(i) - output(k)) * vih(j) * Zh(j) * (1 - Zh(j));
            end

            %The values of the matrix Vih are being updated below
            for k = 1:nOut
                for l = 1:noH
                    vih(k, l) = vih(k, l) + delVih(k) * Zh(l);
                end

                vih(k, l + 1) = vih(k, l + 1) + delVih(k) * 1;
                
            end
            %The values of the matrix Whj are being updated below
            for j = 1:noH

                
                for hid_ = 1:inpN
                    whj(j,hid_) = whj(j,hid_)+delWhj(j) .* x(hid_,i);
                end
                
                whj(j,hid_+1) = whj(j,hid_+1)+1*delWhj(j);
                
            end
        end
        
        %Training
        h = logsig(whj * [x; ones(1, N)]);
        
        y = vih * [h; ones(1, N)];
        
        %Training data error being calculated for each epoch
        err = t-y;
        
        sum_ = sum(err.^ 2) / N;
        
        error = [error sum_];
        
        
        %Validation
        hVal = logsig(whj * [xVal; ones(1, N)]);
        
        yVal = vih * [hVal; ones(1, N)];
        
        %Validation data error being calculated
        err = tVal-yVal;
        
        sum_ =  sum(err.^ 2) / N;
        
        errorVal = [errorVal sum_];
    end
    
    %Training error
    errorAll(1, hiddNo) = error(end);
    %Validation error
    errorAll(2, hiddNo) = errorVal(end);
    
    %PLOTS
    figure();
    
    x_ = linspace(min(xt), max(xt), 100);
    
    len = length(x_);
    h_ = logsig(whj * [x_; ones(1, len)]);
    

    
    y_ = vih * [h_; ones(1,len)];
    %Underlying function getting drawn through the below code
    plot(x_, y_, '-');
    hold on;
    %Data are shown through the symbol '+' on the plot
    plot(xt, rt, '+');
    hold on;
    %The hyperplanes of the hidden unit weights on the first layer
    for hLine = 1:size(whj, 1)
        plot(x_, whj(hLine, :) * [x_; ones(1, len)], '.');
        hold on;
    end
    str = strcat(int2str(noH), ' hidden units');
    title(str);
    hold off;
    
    
    figure();
    plot(x_, y_, '-');
    hold on;
    plot(xt, rt, '+');
    hold on;
    %Hidden unit outputs
    for hLine = 1:size(h_, 1)
        plot(x_, h_(hLine, :), '.');
        hold on;
    end
    title(str);
    hold off;
    figure();
    
    plot(x_, y_, '-');
    hold on;
    plot(xt, rt, '+');
    hold on;
    %Hidden unit outputs multiplied by the weights on the second layer
    for hLine = 1:size(h_, 1)
        plot(x_, h_(hLine, :) * vih(hLine), '.');
        hold on;
    end
    title(str);
    hold off;
    figure();
    

    xAxis = 1:length(error);
    %Training error plotted
    plot(xAxis, error, '-')
    
    hold on;
    %Validation error plotted
    plot(xAxis, errorVal, '.r')
    
    xlabel('Training Epochs');
    ylabel('Mean Square Error');
    title(str);
    legend('Training', 'Validation');
    hold off;
end

fprintf('Errors are shown below (1st row = Training, 2nd row = Validation, columns represent different hidden unit numbers):');
errorAll
