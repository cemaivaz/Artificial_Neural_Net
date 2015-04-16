function [err, r] = TestMixtureOfExperts(problemType, vx, vr, v, m, s)
    if strcmpi(problemType, 'regression')
        ptype = 1;
    else
        ptype = 2;
    end
    outputCount = size(vr, 2);
    expertCount = size(m,1);
    sampleCount = size(vx, 1);
    vx = [ones(sampleCount, 1) vx];
    w = zeros(outputCount, expertCount, sampleCount);
    for j=1:outputCount
        w(j, :, :) = (v(:,:,j)*vx');
    end
    g = (exp(m*vx'));
    
    % 5 25 = size
    %g = (exp(-(sum(vx - m, 2)) .^ 2))';
    
%     sz_ = size(vx, 1);
%     g = zeros(size(m, 1), size(vx, 1));
%     for ex = 1:expertCount
%         for sz = 1:sz_
%             g(ex, sz) = exp(-(vx(sz) - m(ex)) .^ 2 / (2 * s(ex) .^ 2));
%         end
%     end
    
%     g = zeros(size(m, 1), size(vx, 1));
%     for u_ = 1:size(m, 1)
%         for q_ = 1: size(vx, 1)
%            g(u_, q_) = exp(-(sum((vx(q_) - m(u_)), 2) .^ 2)); 
%         end
%     end
    
    g = g ./ repmat(sum(g, 1), expertCount, 1);
    y = zeros(sampleCount, outputCount);
    for i=1:sampleCount
        y(i, :) = ( w(:,:,i) * g(:,i));        
    end
    
    if ptype == 1
        err = sum((y - vr).^2) ./ sampleCount;
        r = y;
    else
        y = exp(y);
        cr = (y==repmat(max(y, [], 2), 1, outputCount));
        err = sum(sum(cr~=vr)) ./ (sampleCount .* outputCount);
        r = cr;
    end    
end