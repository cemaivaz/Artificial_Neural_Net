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


neg_ = 0;
pos_ = 0;

for i = 1:size(valsAll_, 1)
    diff_ = zeros(size(valsAll_, 1) - 1);
    diffInd_ = zeros(size(valsAll_, 1) - 1);
    ind_ = 1;
    min_ = Inf;
    for j = 1:size(valsAll_, 1)
        if i ~= j
            diff = sum((valsAll_(i, :) - valsAll_(j, :)) .^ 2);
            diff_(j) = diff;
            
        end

    end
    diffInd_ = 1:size(valsAll_, 1);
    diffInd_(diffInd_ == i) = [];
    [sorted indices] = sort(diff_);
    
    exclLabels = movementLabels_(indices);
    
    exclLabels = exclLabels(1:3);
    
    maxLabel = mode(exclLabels);
    if maxLabel == movementLabels_(i)
        pos_ = pos_ + 1;
        
    else
        neg_ = neg_ + 1;
    end
    
    %     if ind_ > floor(size(valsAll_, 1) / 2)
    %         pos_ = pos_ + 1;
%     else
%         neg_ = neg_ + 1;
%     end
%     i
%     pos_
%     neg_
end

fprintf('Success rate: %.2f%%', pos_ / (pos_ + neg_));