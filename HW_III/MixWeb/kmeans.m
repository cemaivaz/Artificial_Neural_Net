function [Mhj, Sh] = kmeans(hiddNo)

format long

%Hidden unit numbers are shown below
NH = [2; 4; 8];

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

noH = hiddNo;

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
    allPoi = allPoi_(randOrd, :);
    
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

