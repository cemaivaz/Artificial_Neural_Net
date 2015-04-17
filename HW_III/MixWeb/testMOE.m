function [error_, r] = testMOE(valX, valR, v, mean_, s)

dimenOut = size(valR, 2);
exp_ = size(mean_,1);
dataSize = size(valX, 1);
valX = [ones(dataSize, 1) valX];
wh = ones(dimenOut, exp_, dataSize);
for j=1:dimenOut
    wh(j, :, :) = (v(:,:,j)*valX');
end
gh = (exp(mean_*valX'));


gh = gh ./ repmat(sum(gh, 1), exp_, 1);
y = ones(dataSize, dimenOut);
for i=1:dataSize
    y(i, :) = ( wh(:,:,i) * gh(:,i));
end


error_ = sum((y - valR).^2) ./ dataSize;
r = y;

end