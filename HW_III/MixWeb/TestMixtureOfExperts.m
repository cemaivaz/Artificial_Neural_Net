function [err, r] = TestMixtureOfExperts(problemType, vx, vr, v, m, s)

outputCount = size(vr, 2);
expertCount = size(m,1);
sampleCount = size(vx, 1);
vx = [ones(sampleCount, 1) vx];
w = zeros(outputCount, expertCount, sampleCount);
for j=1:outputCount
    w(j, :, :) = (v(:,:,j)*vx');
end
g = (exp(m*vx'));


g = g ./ repmat(sum(g, 1), expertCount, 1);
y = zeros(sampleCount, outputCount);
for i=1:sampleCount
    y(i, :) = ( w(:,:,i) * g(:,i));
end


err = sum((y - vr).^2) ./ sampleCount;
r = y;

end