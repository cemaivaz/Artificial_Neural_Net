clear
close
clc

file_ = 1;



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



foldNo = 5;
%The below built-in function helps us leverage the cross-validation method,
%where the "k" (fold) value in this case is 10
cv_ = cvpartition(movementLabels_, 'k', foldNo);



dataLength = length(valsAll_);
cntVal = 0;
succ = 0;
fprintf('kNN model is being trained..');
for j = 1:cv_.NumTestSets
    
    trInd = cv_.training(j);
    testInd = cv_.test(j);
    
    
    trDat = valsAll_(trInd, :);
    testDat = valsAll_(testInd, :);
    
    trLabels = movementLabels_(trInd);
    testLabels = movementLabels_(testInd);
    
    
    diff_ = zeros(size(trDat, 1));
    diffInd_ = zeros(size(trDat, 1));
    
    neg_ = 0;
    pos_ = 0;
    
    %%
    %All training data is shown below
    
    
    %Below do we find the eigenvectors for PCA

    trPCAmatr = trDat;
    trPCAmatr = trDat - repmat(mean(trPCAmatr), size(trPCAmatr, 1), 1);
    covAll = cov(trPCAmatr);
    
    [eigenVec, eigenVal] = eig(covAll);
    
    %We try to get the eigenvectors having highest eigenvalues below
    [Ei, ord] = sort(diag(eigenVal));
    eigenVec = eigenVec(:,ord);
    
    
    
    %We get the biggest eigenvectors below
    dimNo = 8;
    eigenVec = eigenVec(:,end:-1:(end - dimNo));


    %%
    

    trDat = trDat * eigenVec;
    
    for i = 1:length(testDat)

        
        
        ind_ = 1;
        max_ = -1;
        testDatPCA = testDat(i, :) * eigenVec;
        testCos = sum(testDatPCA .^ 2) .^ .5;
        for k = 1:size(trDat, 1)

            trCos = sum(trDat(i, :) .^ 2) .^ .5;
            if i ~= k
                diff = sum(testDatPCA .* trDat(k, :)) / (testCos * trCos);
                
                diff_(k) = diff;
                
            end
            
        end
        diffInd_ = 1:size(trDat, 1);
        diffInd_(diffInd_ == i) = [];
        [sorted, indices] = sort(diff_);
        
        exclLabels = trLabels(indices);
        
        exclLabels = exclLabels(1:3);
        
        maxLabel = mode(exclLabels);
        if maxLabel == testLabels(i)
            pos_ = pos_ + 1;
            
        else
            neg_ = neg_ + 1;
        end
        i
        neg_
        pos_
    end
    
    
    succ = succ + pos_ / (pos_ + neg_)
    
    
    
    
    cntVal = cntVal + 1;
end



fprintf('Success rate: %.2f%%', succ / cntVal);