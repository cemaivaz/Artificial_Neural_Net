
%Cem Rifki Aydin    2013800054
%CmpE545    Hw 3
%27.03.2015


close all;
clear all;
clc

format long

%Hidden unit numbers are shown below
NH = [8];

%The number of epochs is shown below
thresh = 100;

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

%%%

%%%

for hiddNo = 1:length(NH)
    noH = NH(hiddNo);
    %%%
    
    
    
    rands = randperm(N);
    
    rands = rands(1:noH);
    
    Hi_ = noH;
    randOrdP = randperm(N);
    randOrdP = randOrd(1:Hi_);
    
    
    allPoi_ = [xt rt];
    
    mi = allPoi_(randOrdP', 1);
    
    
    nCl = 0.058;
    
    miTmp = mi;
    cnt = 0;
    thr = 100;
    t = [];
    
    while cnt < thr
        
        cnt = cnt + 1;
        randOrd = randperm(size(allPoi_, 1))';
        t = allPoi_(randOrd, 2);
        allPoi = allPoi_(randOrd, 1);
        for ith = 1:size(allPoi, 1)
            xt_ = allPoi(ith, 1);
            min_ = Inf;
            minInd = -1;
            for jth = 1:size(mi, 1)
                eucl = sum((xt_ - mi(jth, 1)) .^ 2) ^ .5;
                if eucl < min_
                    min_ = eucl;
                    minInd = jth;
                end
            end
            mi(minInd, 1) = mi(minInd, 1) + nCl * (xt_ - mi(minInd, 1));
            
        end
        nCl = nCl * 0.55;
        if sum((miTmp - mi) .^ 2) < 0.000001
            
            break;
        end
        miTmp = mi;
    end
    
    sh = zeros(1, Hi_);
    clPert = zeros(1, size(allPoi, 1));
    for yth = 1:size(allPoi, 1)
        xt_ = allPoi(yth, 1);
        min_ = Inf;
        minInd = -1;
        for zth = 1:size(mi, 1)
            eucl = sum((xt_ - mi(zth, 1)) .^ 2) .^ .5 ;
            if eucl < min_
                min_ = eucl;
                minInd = zth;
            end
        end
        
        clPert(yth) = minInd;
    end
    
    
    for yth = 1:Hi_
        clPoints = allPoi(clPert == yth, 1);
        
        meanVal = mi(yth, 1);
        
        max_ = -Inf;
        maxInd = -1;
        for zth = 1:size(clPoints, 1)
            clPoint = clPoints(zth, 1);
            diff = sum((meanVal - clPoint) .^ 2) .^ 0.5;
            if diff > max_
                max_ = diff;
                maxInd = zth;
            end
        end
        sh(yth) = max_ / 2;
    end
    %%%
    
    %Whj and Vih matrices are filled with random values
    whj = 0.02*randn(noH,inpN+1) - 0.01;
    
    %upd
    
    %sh(sh == 0) = min(sh(sh ~= 0)) / 1.88;
    
    %sh(sh ~= 0) = sh(sh ~= 0) * 2
    sh(sh == 0) = (max(xt) - min(rt)) / length(xt) * 1.88;
    %sh = sh * 12;

    Mhj = mi;
    Sh = sh;
    
    
    
    vih = 0.02*randn(noH,inpN + 1) - 0.01;
    
    
    
    %upd
    Wih = 0.1*randn(nOut, noH + 1) - 0.05;
    
    Wih_ = 0.1*randn(nOut, noH) - 0.05;
    
    
    x = allPoi(:, 1)';
    %t = allPoi(:, 2)';
    
    n = 0.112; %learning parameter
    
    
    error = []; %Error for training data
    
    errorVal = []; % Error for validation data
    
    Ph = zeros(1, Hi_);
    
    Gh = zeros(size(xt, 1), Hi_);
    
    for c = 0:1:thresh

        for i = 1:N
            firstX = allPoi(i, 1);
            secX = t(i, 1);
                    
           
            for j = 1:noH
                
%                 befSigm(j) = whj(j,1:end-1) * x (:,i) + whj(j,end);
%                 
%                 %Sigmoid value being calculated below
%                 Ph(j) = 1./(1+exp(-befSigm(j)));
                
                
                %UPD
                Ph(j) = exp(-sum(([allPoi(i, 1)] - Mhj(j, :)) .^ 2) / (2 * sh(j) ^ 2));
                
               
            end
            
            for u_ = 1:noH
                Gh(i, u_) = Ph(u_) / sum(Ph);
            end
            
           
            for u_ = 1:noH
                wih(u_) = vih(u_, :) * [firstX; 1];
            end
            %The regression output value being calculated below
            for k = 1:nOut
             
                
                output_(k) = Gh(i, :) * wih';
                
                delVih(:, 1) = n * (secX- output_(k)) * Gh(i, :)' * firstX;
                delVih(:, 2) = n * (secX - output_(k));
                
                vih = vih + delVih;
            end
            %DeltaWhj values for backpropagation being calculated below
            for j = 1:noH
                
                
                delMhj_(j, :) = n * 0.07 * ((secX - output_(k)) * (wih(j) - output_(k)) * Gh(i, j) *  firstX);
                delSh_(j) = n * 0.07 * ((secX - output_(k)) * (wih(j) - output_(k)) * Gh(i, j)  * firstX);
                
                delSh_(j) = 0;
            end
            

            for j = 1:noH
                
              
                tmpMhj = Mhj(j) + delMhj_(j, :);
                if tmpMhj <= min(xt)
                    Mhj(j) = min(xt) + (max(xt) - min(xt)) / (Hi_ * 2);
                elseif tmpMhj >= max(xt)
                    Mhj(j) = max(xt) - (max(xt) - min(xt)) / (Hi_ * 2);
                else
                    Mhj(j) =tmpMhj;
                end
                
            end
        end
        
        h = zeros(length(Sh), size(x, 2));
        
        for wth = 1:length(Sh)
            %h(wth, :) = exp(-sum((allPoi - repmat(Mhj(wth, :), size(allPoi, 1), 1)) .^ 2) .^ .5 ./(2 * Sh(wth) ^ 2));
            
            for sth = 1:size(allPoi, 1)
                h(wth, sth) = exp(-sum((allPoi(sth) - Mhj(wth)) .^ 2) / (2 * Sh(wth) ^ 2));
            end
        end
        
        y = Wih_ * [h];
        
        y = wih * Gh';
        
        
        
        
        
        
        %Training data error being calculated for each epoch

        err = t'-y;
        
        sum_ = sum(err.^ 2) / N;
        
        error = [error sum_];
        
        
        %Validation

        
        %Upd
        hVal = zeros(length(Sh), size(x, 2));
        
        for wth = 1:length(Sh)
            %h(wth, :) = exp(-sum((allPoi - repmat(Mhj(wth, :), size(allPoi, 1), 1)) .^ 2) .^ .5 ./(2 * Sh(wth) ^ 2));
            
            for sth = 1:length(xVal)
                h(wth, sth) = exp(-sum((xVal(sth) - Mhj(wth)) .^ 2) / (2 * Sh(wth) ^ 2));
            end
        end
        
        yVal_ = Wih * [h; ones(1, N)];
        
        
        %Training data error being calculated for each epoch

        
        err = tVal - yVal_;
        
        sum_ = sum(err.^ 2) / N;
        
        errorVal = [errorVal sum_];
        
        %%%
        n = n * 0.73;
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
    
    
    Gh_ = zeros(size(x_, 2), Hi_);
    for p_ = 1:size(x_,2)
        summ = 0;
        for r_ = 1:Hi_
            summ = summ + exp(-sum((x_(p_) - Mhj(r_)) .^ 2)/ (2 * Sh(r_) ^ 2));
        end
        for r_ = 1:Hi_
            Gh_(p_,r_) =  exp(-sum((x_(p_) - Mhj(r_)) .^ 2)/ (2 * Sh(r_) ^ 2)) / summ; 
        end
    end
    
    wih_ = vih * [x_; ones(1, size(x_, 2))];
    
    y_ = sum(Gh_ .* wih_', 2);
    
    
    %UPD
    
    h = zeros(length(Sh), size(x_, 2));
    
    for wth = 1:length(Sh)
        %h(wth, :) = exp(-sum((allPoi - repmat(Mhj(wth, :), size(allPoi, 1), 1)) .^ 2) .^ .5 ./(2 * Sh(wth) ^ 2));
        
        for sth = 1:size(x_, 2)
            h(wth, sth) = exp(-sum((repmat(x_(sth), 1, 2) - Mhj(wth, :)) .^ 2) / (2 * Sh(wth) ^ 2));
        end
    end
    
    %y_ = Wih_ * h;
    
    %Underlying function getting drawn through the below code
    plot(x_, y_, '-');
    hold on;
    %Data are shown through the symbol '+' on the plot
    plot(xt, rt, '+');
    hold on;

    str = strcat(int2str(noH), ' hidden units');
    title(str);
    hold off;
    
    
    figure();
    plot(x_, y_, '-');
    hold on;
    plot(xt, rt, '+');
    hold on;

    title(str);
    hold off;
    figure();
    
    plot(x_, y_, '-');
    hold on;
    plot(xt, rt, '+');
    hold on;

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



