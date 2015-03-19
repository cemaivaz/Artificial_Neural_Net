%Cem Rifki Aydin    2013800054
%CmpE545 Hw 1
%In this homework, logistic discriminant algorithm is implemented

%Clear console, and the past data
clear
clc


%Read the data in the test file
fileTest = textread('test.txt', '%s');


oneDig = [];


%The map below is used for keeping the data (of 256-dimensional size) for
%each digit class
mapTest_ = containers.Map('keyType', 'char', 'valueType', 'any');

%One can change the learning parameter however s/he wants. The smaller the
%value, the more slowly it converges however
learnPar = 0.07;

for i = 1:size(fileTest, 1)
    %Below reside the digit class labels in the file at a specific row, the
    %other lines including the representations (of 256-dimensions) thereof
    if (mod(i, 17) == 0)
        
        val = char(fileTest(i, :));
        
        tmp = [];
        if isKey(mapTest_, val)
            tmp = mapTest_(val);
        end
        
        mapTest_(val) = [tmp {oneDig}];
        
        
        oneDig = [];
    else
        arr = [];
        ch = char(fileTest(i, :));
        for k = 1:length(ch)
            arr = [arr str2double(ch(k))];
        end
        oneDig = [oneDig arr];
    end
end

mapLength = length(keys(mapTest_));

allKeys = keys(mapTest_);



%Now is the training data being read
fileTr = textread('train.txt', '%s');


oneDig = [];

%Another map for containing the key-value pairs of the training data
map_ = containers.Map('keyType', 'char', 'valueType', 'any');

fprintf('Different digit class models being trained..\n');

arrDigs = [];

allDig = [];

%The process that had been performed for the test data is now going to be
%done for the training data in the same way
for i = 1:size(fileTr, 1)
    if (mod(i, 17) == 0)
        
        
        val = char(fileTr(i, :));
        
        
        allDig = [allDig; val];
        
        
        tmp = [];
        if isKey(map_, val)
            tmp = map_(val);
        end
        len = oneDig;
        map_(val) = [tmp {oneDig}];
        
        arrDigs= [arrDigs; {map_}];
        oneDig = [];
        map_ = containers.Map('keyType', 'char', 'valueType', 'any');
    else
        arr = [];
        ch = char(fileTr(i, :));
        for k = 1:length(ch)
            arr = [arr str2double(ch(k))];
        end
        oneDig = [oneDig arr];
        
    end
end


%Unique class labels
allDig = unique(allDig);
%A specific class label, be it '0', or '1', etc.
aDig = allDig(1);

%Dimension number. (In this case, it is 256)
dimenNo = length(len);



%The number of classes
digNo = length(allDig);

%The Wij matrix
wij = zeros(digNo, dimenNo);


%First, the matrix Wij is filled with numbers randomly produced
for i = 1:digNo
    for j = 0:dimenNo
        wij(i, j + 1) = rand() * 0.02 - 0.01;
    end
end

allKeys = allDig;

%The below parameters are going to be used for detecting whether the
%convergence is met or not
loopNo = 0;
summConv = Inf;
convFact = 1e-03;

oi = zeros(1, digNo);

%The training data are shuffled, since accessing the data randomly is what
%is a reasonable approach in order to overcome bias
arrDigs = arrDigs(randperm(length(arrDigs)), :);

%Confusion matrix
confMatrTr = zeros(digNo, digNo);

%The below variable helps find the overall success rate obtained after all the epochs elapsed for
%training data
trEpochSucc = 0;


%The below variable helps find the overall success rate obtained after all the epochs elapsed for
%test data
testEpochSucc = 0;

while loopNo < 100 && summConv > convFact
    loopNo = loopNo + 1;
    %The matrix delta-Wij is updated below (all of its elements are updated as zero)
    deltaWij = zeros(digNo, dimenNo + 1);
    
    wijComp = wij;
    
    for q = 1: length(arrDigs)
        
        map_ = arrDigs{q};
        tmpHash = keys(map_);
        classTmp = map_(tmpHash{1});
        
        
        for u = 1:length(classTmp)
            
            
            subClassTmp= (classTmp(u));
            %The vector Oi gets updated below
            for j = 1:digNo
                oi(j) = 0;
                for k = 0:dimenNo
                    
                    if k == 0
                        oi(j) = oi(j) + wij(j, k + 1);
                    else
                        oi(j) = oi(j) + wij(j, k + 1) .* subClassTmp{1}(k);
                    end
                    
                end
            end
           
            summ = 0;
            %The summation is calculated by summing up all the Oi values (by
            %taking the exponential values thereof)
            for y = 1:digNo
                summ = summ + exp(oi(y));
            end
            
            %Normalization is performed
            for j = 1:length(allDig)
                
                yi(j) = exp(oi(j)) ./ summ;
                
                
            end
            var rit;
            
            for j = 1:length(allDig)
                
                %Below is examined the fact whether the data being
                %processed belongs to the label it is expected to be
                if char(tmpHash{1}) == char(allDig(j))
                    
                    rit = 1;
                else
                    
                    rit = 0;
                end
                %Delta-Wij is being updated below, in accordance with the 
                %stochastic gradient descent algorithm
                for k = 0:dimenNo
                    
                    if k == 0
                        xjt = 1;
                    else
                        xjt = subClassTmp{1}(k);
                    end
                   
                    deltaWij(j, k + 1) = deltaWij(j, k + 1) + (rit - yi(j)) *  xjt;
                    
                end
                
            end
        end
        
        
        
    end
    
    %The learning parameter helps find the local minima
    for i = 1:digNo
        for j = 0:dimenNo
            wij(i, j + 1) = wij(i, j + 1) + learnPar * deltaWij(i, j + 1);
        end
        
    end
    
    
    %The value below helps us realize whether the convergence is met or not
    summConv = sum(sum((wij - wijComp) .^ 2));
    %The learning parameter decreases gradually at each epoch
    learnPar = learnPar * 0.88;
    
    
    confMatrTmp = zeros(digNo, digNo);
    cnterAll= 0;
    succ = 0;
    
    %The loop below helps find the success rate at each epoch for training
    %data
    for q = 1: length(arrDigs)
        
        map_ = arrDigs{q};
        tmpHash = keys(map_);
        tmpHash = tmpHash{1};
        classTmp = map_(tmpHash);
        
        cnterAll = cnterAll + 1;
        
        for u = 1:length(classTmp)
            
            
            subClassTmp= (classTmp(u));
            subClassTmp = subClassTmp{1};
            
            max = -Inf;
            maxInd = 1;
            for k = 1:size(wij)
                res = sum(wij(k, 2:dimenNo + 1) .* subClassTmp) + wij(k, 1);
                if res > max
                    max = res;
                    maxInd = k;
                end
            end
            
            if (maxInd == str2double(tmpHash) + 1)
                
                succ = succ + 1;
            end
            
            confMatrTmp(str2double(tmpHash) + 1, maxInd ) = confMatrTmp(str2double(tmpHash) + 1, maxInd) + 1;
        end
    end
    
    trEpochSucc = trEpochSucc + succ / cnterAll * 100;
    
    fprintf('___\n\n(TRAINING DATA) Run (epoch) %i, succ. rate: %0.2f%%\n', loopNo, succ / cnterAll * 100);
    Confusion_Matrix_Tr = confMatrTmp;
    
    
    
    succ = 0;
    cntAll = 0;
    
    
    
    confMatr = zeros(digNo, digNo);
    %The below operations help find the success rate at each epoch for test
    %data
    for i = 1:mapLength
        dig = mapTest_(char(allKeys(i)))';
        for j = 1:length(dig)
            cntAll = cntAll + 1;
            testDig = dig(j);
            max = -Inf;
            maxInd = 1;
            for k = 1:size(wij)
                res = sum(wij(k, 2:dimenNo + 1) .* cell2mat(testDig)) + wij(k, 1);
                if res > max
                    max = res;
                    maxInd = k;
                end
            end
            if (maxInd == i)
                
                succ = succ + 1;
            end
            confMatr(i, maxInd ) = confMatr(i, maxInd) + 1;
        end
    end
    
    testEpochSucc = testEpochSucc + succ / cntAll * 100;
   
    
    %Success rate gets printed
    fprintf('_____\n(TEST DATA) Run (epoch) %i, succ. rate: %0.2f%%\n', loopNo, succ / cntAll * 100);
    
    Confusion_Matrix_Test = confMatr
    
    
    
    
    
    
    
end

%The overall success rate gets printed below for the training data
fprintf('\nThe success rate for the training data (based on the epochs): %0.2f%%\n\n\n', trEpochSucc / loopNo);


%The overall success rate gets printed below for the test data
fprintf('\nThe success rate for the test data (based on the epochs): %0.2f%%\n\n\n', testEpochSucc / loopNo);


