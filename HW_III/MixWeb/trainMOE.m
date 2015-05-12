function [v, mh, s] = trainMOE( dataX, dataR, exp_, iterLim, learnParam, decaying)
%exp_: The number of experts

dataSize = length(dataX);
dimenOut = size(dataR, 2);
dimen = 1 + size(dataX,2);
  


v = (rand(exp_,dimen, dimenOut)*0.02)-0.01;

%The m, and s values are produced through the k-means algorithm
[mh, s] = kmeans(exp_);

mh = [rand(exp_, dimen - 1) * 0.02 - 0.01 mh]


%Below, the bias unit is taken into account
dataX = [ones(dataSize,1) dataX];

iter_ = 1;
errs = zeros(iterLim, 1);
prevErr = Inf;
while true
    %Samples are randomly shuffled
    rand_ = randperm(dataSize);
    for i=1:dataSize
        k = rand_(i);
        xt = dataX(k,:);
        rt = dataR(k,:);
        
       
        gh = exp(mh * xt')';
        
        summ_ = sum(gh);
        gh = gh ./ summ_;
        
        w = zeros(dimenOut, exp_);
        
        
        for j=1:dimenOut
            w(j,:) = (v(:,:,j)*xt')';
        end     
        
        %Output
        yi = (w*gh')';
        %Delta m, and v are calculated
        for r=1:exp_
            for j=1:dimenOut
                deltaV = learnParam .* ( rt(1,j) - yi(1,j)) * gh(1, r) * xt;

                v(r, :, j) = v(r, :, j) + deltaV;
                
            end

            deltaM = learnParam .* sum((rt - yi) .* (w(:, r)' - yi)) .* gh(1, r) * xt;

            mh( r, : ) = mh( r, : ) + deltaM;
            
        end        
    end
   
      
    %Training error
    err = testMOE( dataX(:,2:dimen), dataR, v, mh, s);
    fprintf('Err: %f\n', err);
    errs(iter_, 1) = err;
    
    iter_ = iter_ + 1;    

    %Break if the maximum iteration number is reached, or error is quite
    %small
    if err < 0.00001 || iter_ > iterLim
       
        break;
    end
    learnParam = decaying * learnParam;
  
    %Break if convergence is met
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

    gh = (exp(mh*xt'))';
    
    gh = gh ./ sum(gh);
    w = ones(dimenOut, exp_);
    
    for j=1:dimenOut
        w(j,:) = (v(:,:,j)*xt')';
        
    end
    
    w_ = [w_; w];

    
    g_ = [g_; gh];
    
    %Output
    yi = (w*gh')';
    yi_ = [yi_ yi];
    
end

%Plots are drawn below
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