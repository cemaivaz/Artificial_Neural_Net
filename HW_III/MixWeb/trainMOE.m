function [v, mh, s] = trainMOE( dataX, dataR, exp_, iterLim, learnParam, decaying)


dataSize = size(dataX,1);
dimen = size(dataX,2) + 1;
dimenOut = size(dataR, 2);


% add bias unit to x
dataX = [ones(dataSize,1) dataX];



% learnParam = 0.1;
% decaying = 0.98;

% initialize parameters
% use first points for mh
% mh = dataX(1:exp_,:);
mh = (rand(exp_, dimen)*0.02)-0.01;
% v in {-0.01, 0.01}
v = (rand(exp_,dimen, dimenOut)*0.02)-0.01;


[mh, s] = kmeans(exp_);
% % mh = [repmat(size())];
%mh = [repmat(1, size(mh, 1), 1) mh];

mh = [rand(exp_, dimen - 1) * 0.02 - 0.01 mh]
%mh = [rand(exp_, 1) * 0.02 - 0.01 mh];
iter_ = 1;
errs = zeros(iterLim, 1);
prevErr = Inf;
while 1
    % choose next training instance randomly
    rand_ = randperm(dataSize);
    for i=1:dataSize
        k = rand_(i);
        xt = dataX(k,:);
        rt = dataR(k,:);
        
       
        % calculate intermediate values
        gh = (exp(mh*xt'))';
        
        gh = gh ./ sum(gh);
        w = zeros(dimenOut, exp_);
        for j=1:dimenOut
            w(j,:) = (v(:,:,j)*xt')';
        end     
        
        % calculate output
        
        % output for each output dimension
        yi = (w*gh')';
        % calculate delta v and mh for each output unit
        for r=1:exp_
            for j=1:dimenOut
                deltaV = learnParam .* ( rt(1,j) - yi(1,j)) * gh(1, r) * xt;

                v(r, :, j) = v(r, :, j) + deltaV;
                
            end
            % calculate delta mh
            deltaM = learnParam .* sum((rt - yi) .* (w(:, r)' - yi)) .* gh(1, r) * xt;

            mh( r, : ) = mh( r, : ) + deltaM;
            
        end        
    end
   
    learnParam = learnParam * decaying;
    % calculate training set error

    err = testMOE( dataX(:,2:dimen), dataR, v, mh, s);
    fprintf('Err: %f\n', err);
    errs(iter_, 1) = err;
    
    iter_ = iter_ + 1;    
    % check stop condition
    if iter_ > iterLim 
        
        break;
    end
    if err < 0.00001
       
        break;
    end
%     if abs(prevErr - err) < 0.0001
%         break;
%     end
%     prevErr = err;
end

x_ = min(dataX(:, 2)):0.01:max(dataX(:, 2));
x_ = x_';

x_ = [ones(size(x_, 1),1) x_];
yi_ = [];
g_ = [];
w_ = [];
for i_ = 1:size(x_, 1)
    
    xt = x_(i_, :);
    % calculate intermediate values
    gh = (exp(mh*xt'))';
    
    gh = gh ./ sum(gh);
    w = ones(dimenOut, exp_);
    
    for j=1:dimenOut
        w(j,:) = (v(:,:,j)*xt')';
        
    end
    
    w_ = [w_; w];
    % calculate output
    
    
    g_ = [g_; gh];
    
    % output for each output dimension
    yi = (w*gh')';
    yi_ = [yi_ yi];
    
end

size(w_)
figure()
plot(x_(:, 2)', yi_, '-');
hold on;
plot(dataX(:, 2), dataR, '+');


figure()
plot(x_(:, 2)', yi_, '-');
hold on;
plot(dataX(:, 2), dataR, '+');
hold on;
for i_ = 1:exp_
    plot(x_(:, 2)', w_(:, i_), '- ');
end

size(x_)
size(g_)
figure()
plot(x_(:, 2)', yi_, '-');
hold on;
plot(dataX(:, 2), dataR, '+');
hold on;
for i_ = 1:exp_
    plot(x_(:, 2)', g_(:, i_), '- ');
end

end