size_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigram.txt');% 60.75%
movementLabels_ =  dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigram.txt', ',', [0 0 size(size_, 1) - 1 0]);
valsAll_ = dlmread('C:\\Users\\asus\\workspace\\Sentiment_Analysis\\outputBigram.txt', ',', [0 1 size(size_, 1) - 1 size(size_, 2) - 1]);

neg_ = 0;
pos_ = 0;
for i = floor(size(valsAll_, 1) / 2):size(valsAll_, 1)
    ind_ = 1;
    min_ = Inf;
    for j = 1:size(valsAll_, 1)
        diff = sum((valsAll_(i, :) - valsAll_(j, :)) .^ 2);
        if diff < min_
            ind_ = j;
            min_ = diff;
        end
    end
    if ind_ > floor(size(valsAll_, 1) / 2)
        pos_ = pos_ + 1;
    else
        neg_ = neg_ + 1;
    end
    i
    pos_
    neg_
end

neg_
pos_