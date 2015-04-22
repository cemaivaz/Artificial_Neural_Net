%"WEARABLE COMPUTING" PROJECT - Team 11, 13/12/2014
%DEVELOPING EXOSKELETON SYSTEM FOR THE MOBILITY-IMPAIRED
%Ali Ozcan, Bugra Oral, Erdem Emekligil, Onur Satici, Cem Rifki Aydin

%PART 3 - Implementation of SVM (radial basis)

%Below is the source code written for the exoskeleton system - recognition of different
%arm movements.

clear
clc
close all
make 
mex -setup

file_ = 1;

%We scan the files in the 'movements' folder
subDir = dir('movements');

subDirInd = [subDir.isdir];

subDir_ = {subDir(subDirInd).name};
ind = ~ismember(subDir_, {'.', '..'});


testDataAll = [];
movementLabels = [];


valsAll = [];

%All the files in the subdirectories of the folder 'movements' are scanned
for direc = find(ind)
    newDir = fullfile('movements', subDir_{direc});
    allFiles = dir(newDir);
    
    x = newDir;
    fileN = {};
    for file = allFiles';
        
        if strcmp(file.name, '.') == 0 && strcmp(file.name, '..') == 0
            fileN = [fileN; char(strcat(strcat('movements\', strcat(subDir_{direc}, '\')), char(file.name)))];
        end
        
    end
    
    allData = cellstr(fileN);
    
    for u = 1:length(allData)
        
        
        fileMv = allData(u);
        fileMv = char(fileMv);
        
        vals_ = dlmread(fileMv, ' ', 0, 0);
        valsAll = [valsAll; {vals_}];
        
        movementLabels = [movementLabels; {subDir_{direc}}];
    end
end


if file_ == 1
    movementLabels_ =  dlmread('output.txt', ',', [0 0 7085 0]);
    valsAll_ = dlmread('output.txt', ',', [0 1 7085 1699]);;
    
else
    movementLabels_ =  dlmread('output2.txt', ',', [0 0 39 0]);
    valsAll_ = dlmread('output2.txt', ',', [0 1 39 1699]);;
    
end


 movementLabels = [];
 valsAll = [];
for i = 1:size(movementLabels_, 1)
    movementLabels = [movementLabels;  (movementLabels_(i, :))];
end
for i = 1:size(valsAll_, 1)
    valsAll = [valsAll; {valsAll_(i, :)}];
    
end
fprintf('_____')

%Unique movement labels are determined
order = unique(movementLabels);
coef = length(movementLabels) / length(order);
length(movementLabels);

foldNo = 5;
%The below built-in function helps us leverage the cross-validation method,
%where the "k" (fold) value in this case is 10
cv_ = cvpartition(movementLabels, 'k', foldNo);



cnterAll = length(movementLabels);
cnterSucc = 0;

avgSucc = 0;

orderInt = 1:length(unique(movementLabels));

avgSucc = 0;

RESULTS = [];
resind = 1;


cntTestFail = 0;
cntTestSucc = 0;
%We, below, iterate over the sets created through cross-validation
for j = 1:cv_.NumTestSets
    
    trDat = cv_.training(j);
    testDat = cv_.test(j);

    testIter = 1;
    
    mod_ = valsAll(trDat == 1);
    
    
    test_ = valsAll(trDat == 0);
    
    labelsTr_ = movementLabels((trDat == 1)', :);
    labelsTest_ = movementLabels((trDat == 0)', :);
    
    
    train_data = cell2mat(mod_);
    train_label = str2double(labelsTr_);
    
    %The below built-in function provided by the libsvm library trains
    %a model through the data collected by the application "Accelerometere Monitor".
    model_linear = svmtrain(labelsTr_, train_data, '-t 2')
    
    
    for w = 1:length(test_)
        
        test_label = labelsTest_(testIter);
        test_data = test_{w};
        
        testIter = testIter + 1;
       
        %The below built-in function detects the labels, and accuracy
        %percentage, where the test data and labels to be classified are inputs.
        [predict_label_L, accuracy_L, dec_values_L] = svmpredict(test_label, test_data, model_linear);
        
        avgSucc = avgSucc + accuracy_L(1);
        
        
        if accuracy_L(1) ~= 100
            cntTestFail = cntTestFail + 1;
            RESULTS{resind} = ['FALSE - Test movement: ', int2str(test_label), ', Predicted movement: ', int2str(mod((test_label + 1), 2))];
        else
            cntTestSucc = cntTestSucc + 1;
            RESULTS{resind} = ['TRUE - Test movement: ',int2str(test_label), ', Predicted movement: ', int2str(test_label)];
        end
        
        resind = resind + 1;
    end
    
    
    
    % Below are the options for the SVM method
    
    %-t 3: sigmoid function
    
    %-t 2: radial basis
    
    %-t 1: polynomial (default degree is 3)
    
    %-t 0: linear
    
    
    
    
    RESULTS{resind} = '_______________';
    resind = resind + 1;
end


fprintf('SVM results:\n\n');

for i = 1:length(RESULTS)
    fprintf('%s\n', RESULTS{i});
end
%Success rate gets printed
fprintf('\n\nOverall success rate: %0.2f%%\n', cntTestSucc / (cntTestSucc + cntTestFail) * 100);


