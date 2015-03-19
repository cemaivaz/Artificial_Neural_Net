
%Cem Rifki Aydin    2013800054
%CmpE545    Hw 2
%13.03.2015

%Clear console, and the past data
clear
clc

%Precision is chosen to be large
format long

fprintf('Model is being trained..\n\n');

%Iteration number
iterNo = 600;


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

%Sorting is also leveraged while plotting the graphs
[s, ord] = sort(xt);



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



xtVal = xtVal(randOrd);
rtVal = rtVal(randOrd);

%Sorting the validation data may be leveraged later on in plotting
[s, ord] = sort(xtVal);


len = length(xt);

lenVal = length(xtVal);


dimenNo = 1;
wh0 = 1;

K = length(rt);

K = 1;

trFluc = [];


%Iteration over different hidden units
for h_ = 1:length(Hidd)
    H = Hidd(h_);
    
    
    %The below matrices are used in measuring the error
    errTr = [];
    errVal = [];
    %First, the matrix Wij is filled with numbers randomly produced
    for h = 1:H
        for j = 0:dimenNo
            whj(h, j + 1) = rand() * 0.02 - 0.01;
        end
    end
    
    %The matrix Vih is being filled with those numbers randomly produced
    for i = 1:K
        for h = 0:H
            vih(i, h + 1) = rand() * 0.02 - 0.01;
        end
    end

    deltaVi = zeros(K, H + 1);
    
    loopNo = 0;
    deltaWh = [];
    
    %Learning parameter coefficient
    learnPar = 0.1;
    while loopNo < iterNo
        %Online learning backpropagation algorithm being run
        for t = 1:length(xt)
            xt_ = [xt(t)];
            
            
            zh = zeros(1, H + 1);
            zh(1) = 1;
            
            
            for h = 1:1:H
                sum_ = 0;
                
                
                for d = 1:dimenNo

                    sum_ = sum_ + whj(h, d + 1) * xt_(d);
                end
                sum_ = sum_ + whj(h, 1);
                
                %The output values of hidden units being calculated
                zh(h + 1) = 1 / (1 + exp(-sum_));
                
            end
            
            zh_ = zh;

            trZh = zh';
            
            for i = 1:K
                %The outputs being updated
                yi(i) = vih(i, :) * trZh;
                
            end
            

            for i = 1:K
                %The delta Vi(h) values being updated
                deltaVi(i, :) = learnPar * (rt(t) - yi(i)) * zh;

            end
            
            
            for h=1:H
                
                
                tmpRt = rt;
                summ_ = 0;
                yiT = yi';
                for i = 1:K
%                     if loopNo == 22
%                         fprintf('___');
%                         tmpRt(t)
%                         yi(i)
%                     end
                   
                    summ_ = summ_ + (tmpRt(t) - yi(i)) * vih(i, h + 1);
                end
                
                %The delta Wh(j) values being calculated
                deltaWh(h) = learnPar * summ_ * zh(h + 1) * (1 - zh(h + 1)) * xt(t);
            end
            
            for i = 1:K
                %The Vih matrix being updated
                vih(i, :) = vih(i, :) + deltaVi(i, :);
            end
            
            %The matrix Whj being updated
            whj = whj + repmat(deltaWh', 1, 2);
            
            
        end
        
        loopNo = loopNo + 1;
        
        %The learning parameter being gradually decreased
        learnPar = learnPar;% * 0.799;
        
        %TRAINING ERROR
        
        yi_ = yi;
        rt_ = rt';
        
        
        %
        zh(1) = 1;
        yiAll = zeros(1, len);
        yiAllVal = zeros(1, len);
        
        zhVal(1) = 1;
        for u = 1:len
            xt_ = xt(u);
            xtVal_ = xtVal(u);
            for h = 1:1:H
                sum_ = 0;
                
                sumVal_ = 0;
                for d = 1:dimenNo
                    
                    sum_ = sum_ + whj(h, d + 1) * xt_(d);
                    
                    sumVal_ = sumVal_ + whj(h, d + 1) * xtVal_(d);
                    
                end
                sum_ = sum_ + whj(h, 1);
                
                sumVal_ = sumVal_ + whj(h, 1);
                
                %The output values of hidden units being calculated
                zh(h + 1) = 1 / (1 + exp(-sum_));
                
                zhVal(h + 1) = 1 / (1 + exp(-sumVal_));
                
            end
            
               
            zh_ = zh;

            trZh = zh';
            
            trZhVal = zhVal';
            
            for i = 1:K
                %The outputs being updated
                yi(i) = vih(i, :) * trZh;
                
                yiVal(i) = vih(i, :) * trZhVal;
                
            end
            yiAll(u) = yi(1);
            
            yiAllVal(u) = yiVal(1);
        end
        %
        
        err = 1/2 * sum((yiAll' - rt) .^ 2);
        
        errTr = [errTr; [err H]];
        
        err = 1/2 * sum((yiAllVal' - rtVal) .^ 2);
        
        errVal = [errVal; [err H]];
        
        %VALIDATION ERROR
%         xtTmp = xt(ord);
%         
%         rtTmp = rt(ord); yiTmp = yi(ord);
%         
%         p = polyfit(xtTmp,yiTmp',9);
%         
%         f = polyval(p,xtVal);
%         
%         
%         errVal = [errVal; [(sum((rtVal - f).^ 2) * 1 / 2) H]];
%         
        %If convergence met, break the loop
        if err < 0.00001
            break;
        end
        
        
        %EXTRA: If one wants to plot the graph with the regression being
        %implemented, s/he would uncomment the below command block
        
        %{
        figure
        plot(xtTmp,yiTmp','o',xtTmp,f,'-')
        title('Parametric reg. / Degree = 9');
        xlabel('x');
        ylabel('y');
        %}
        
        
    end
    
    if H == 2
        trFluc = errTr;
    end
    
    
    %Training error (on average) gets printed (mean-squares)
%     summa_ = sum(errTr, 1);
%     fprintf('The training error for the hidden unit number of %d: %0.22f\n', H, summa_(1)/size(errTr, 1));
    
    %Validation error (on average) gets printed (mean-squares)
%     summa_ = sum(errVal, 1);
%     fprintf('The validation error for the hidden unit number of %d: %0.22f\n\n\n', H, summa_(1)/size(errVal, 1));
    
    
end



%The below vector is leveraged in plotting the graph (The x values)
x_huw = linspace(min(xt), max(xt),100);

lenX = length(x_huw);


%PLOT 1: The hyperplanes of the hidden unit weights on the first layer
figure(1)

%Here, sigmoid values are NOT taken account of
y_huw1 = whj(1, 1) + x_huw * whj(1, 2);

y_huw2 = whj(2, 1) + x_huw * whj(2, 2);

%Uncomment the below so as to see the data, and the underlying sine
%function plotted

plot(xt, rt, '+', x_huw, sin(6 * x_huw), '-')
hold on;

plot(x_huw, y_huw1, '--gs');
hold on;
plot(x_huw, y_huw2, '--rs');
hold off;

%PLOT 2: Hidden unit outputs
figure(2)

y_huo1 = zeros(1, lenX);
y_huo2 = zeros(1, lenX);

for i = 1:lenX
    %Sigmoid values are taken into account
    y_huo1(i) = 1 / (1 + exp(-y_huw1(i)));
    y_huo2(i) = 1 / (1 + exp(-y_huw2(i)));
end

%{
plot(xt, rt, '+', x_huw, sin(6 * x_huw), '-')
hold on;
%}
plot(x_huw, y_huo1', '--gs');
hold on;
plot(x_huw, y_huo2', '--rs');
hold off;

%PLOT 3: hidden unit outputs multiplied by the weights on the second layer
figure(3)
%Below are the weights of the second layer are taken account of
y_how1 = sum(vih(:, 2) * y_huo1);
y_how2 = sum(vih(:, 3) * y_huo2);

%{
plot(xt, rt, '+', x_huw, sin(6 * x_huw), '-')
hold on;
%}
plot(x_huw, y_how1', '--gs');
hold on;
plot(x_huw, y_how2', '--rs');
hold off;


%Fluctuation of the training error in time
figure(4)
xAxis = [];
for i = 1:size(trFluc, 1)
    xAxis = [xAxis; i];
end
plot(xAxis, trFluc(:, 1), '--bs')
