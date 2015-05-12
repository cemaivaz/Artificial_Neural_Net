function error_ = testMOE(valX, valR, v, mean_, s)

dimenOut = size(valR, 2);
exp_ = size(mean_,1);
dataSize = size(valX, 1);
valX = [ones(dataSize, 1) valX];
wh = ones(dimenOut, exp_, dataSize);

j= 1;
while j <= dimenOut
    wh(j, :, :) = v(:,:,j)*valX';
    j = j + 1;
end
gh = exp(mean_*valX');

div = repmat(sum(gh, 1), exp_, 1);
gh = gh ./ div;
y = ones(dataSize, dimenOut);

i = 1;
while i <= dataSize
    y(i, :) = ( wh(:,:,i) * gh(:,i));
    i = i + 1;
end


error_ = sum((y - valR).^2) / dataSize;


end