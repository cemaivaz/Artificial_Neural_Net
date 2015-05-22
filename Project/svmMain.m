%Cem Rýfký Aydýn    2013800054


%Implementation of SVM (radial basis)
%Below is the source code written for the exoskeleton system - recognition of different
%arm movements.

clear
clc
close all
make 
mex -setup

file_ = 4;

%We scan the files in the 'movements' folder



if file_ == 1
    size_ = dlmread('output.txt');
    movementLabels_ =  dlmread('output.txt', ',', [0 0 size(size_, 1) - 1 0]);
    valsAll_ = dlmread('output.txt', ',', [0 1 size(size_, 1) - 1 size(size_, 2) - 1]);
elseif file_ == 2 %bigrams
    size_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigram.txt');% 60.75%
    movementLabels_ =  dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigram.txt', ',', [0 0 size(size_, 1) - 1 0]);
    valsAll_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigram.txt', ',', [0 1 size(size_, 1) - 1 size(size_, 2) - 1]); 
elseif file_ == 3 %unigram - including stopwords
    size_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputStopwords.txt');
    movementLabels_ =  dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputStopwords.txt', ',', [0 0 size(size_, 1) - 1 0]);
    valsAll_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputStopwords.txt', ',', [0 1 size(size_, 1) - 1 size(size_, 2) - 1]);     
elseif file_ == 4 %bigram & including NOT stopwords
    size_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigramStopwordsEl.txt');%60.57%
    movementLabels_ =  dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigramStopwordsEl.txt', ',', [0 0 size(size_, 1) - 1 0]);
    valsAll_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigramStopwordsEl.txt', ',', [0 1 size(size_, 1) - 1 size(size_, 2) - 1]);         
elseif file_ == 5 %Unigram - NOT tfidf
    size_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputFreq.txt');% %
    movementLabels_ =  dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputFreq.txt', ',', [0 0 size(size_, 1) - 1 0]);
    valsAll_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputFreq.txt', ',', [0 1 size(size_, 1) - 1 size(size_, 2) - 1]);      
else
    size_ = dlmread('output2.txt');
    size(size_)
    movementLabels_ =  dlmread('output2.txt', ',', [0 0 size(size_, 1) - 1 0]);
    valsAll_ = dlmread('output2.txt', ',', [0 1 size(size_, 1) - 1 size(size_, 2) - 1]);
    
end


 sentLabels = [];
 valsAll = [];
for i = 1:size(movementLabels_, 1)
    sentLabels = [sentLabels;  (movementLabels_(i, :))];
end
for i = 1:size(valsAll_, 1)
    valsAll = [valsAll; {valsAll_(i, :)}];
    
end
fprintf('_____')

%Unique movement labels are determined
order = unique(sentLabels);
coef = length(sentLabels) / length(order);
length(sentLabels);

foldNo = 5;
%The below built-in function helps us leverage the cross-validation method,
%where the "k" (fold) value in this case is 10
cv_ = cvpartition(sentLabels, 'k', foldNo);



cnterAll = length(sentLabels);
cnterSucc = 0;

avgSucc = 0;

orderInt = 1:length(unique(sentLabels));

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
    
    labelsTr_ = sentLabels((trDat == 1)', :);
    labelsTest_ = sentLabels((trDat == 0)', :);
    
    
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
            RESULTS{resind} = ['FALSE - Sentiment: ', int2str(test_label), ', Predicted sentiment: ', int2str(mod((test_label + 1), 2))];
        else
            cntTestSucc = cntTestSucc + 1;
            RESULTS{resind} = ['TRUE - Sentiment: ',int2str(test_label), ', Predicted sentiment: ', int2str(test_label)];
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


