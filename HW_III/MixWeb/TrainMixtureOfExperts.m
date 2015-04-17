function [v, m, s] = TrainMixtureOfExperts(problemType, moeType, tx, tr, expertCount, maxIter, learningRate, decay)


sampleCount = size(tx,1);
dim = size(tx,2) + 1;
outputCount = size(tr, 2);


% add bias unit to x
tx = [ones(sampleCount,1) tx];



% learningRate = 0.1;
% decay = 0.98;

% initialize parameters
% use first points for m
% m = tx(1:expertCount,:);
m = (rand(expertCount, dim)*0.02)-0.01;
% v in {-0.01, 0.01}
v = (rand(expertCount,dim, outputCount)*0.02)-0.01;


[m, s] = kmeans(expertCount);
% % m = [repmat(size())];
%m = [repmat(1, size(m, 1), 1) m];

m = [rand(expertCount, dim - 1) * 0.02 - 0.01 m]
%m = [rand(expertCount, 1) * 0.02 - 0.01 m];
iters = 1;
errs = zeros(maxIter, 1);
prevErr = Inf;
while 1
    % choose next training instance randomly
    trSeq = randperm(sampleCount);
    for i=1:sampleCount
        k = trSeq(i);
        xt = tx(k,:);
        rt = tr(k,:);
        
       
        % calculate intermediate values
        g = (exp(m*xt'))';
        
        g = g ./ sum(g);
        w = zeros(outputCount, expertCount);
        for j=1:outputCount
            w(j,:) = (v(:,:,j)*xt')';
        end     
        
        % calculate output
        
        % output for each output dimension
        yi = (w*g')';
        % calculate delta v and m for each output unit
        for r=1:expertCount
            for j=1:outputCount
                dv = learningRate .* ( rt(1,j) - yi(1,j)) * g(1, r) * xt;

                v(r, :, j) = v(r, :, j) + dv;
                
            end
            % calculate delta m
            dm = learningRate .* sum((rt - yi) .* (w(:, r)' - yi)) .* g(1, r) * xt;

            m( r, : ) = m( r, : ) + dm;
            
        end        
    end
   
    learningRate = learningRate * decay;
    % calculate training set error

    err = TestMixtureOfExperts(problemType, tx(:,2:dim), tr, v, m, s);
    fprintf('Error: %f\n', err);
    errs(iters, 1) = err;
    
    iters = iters + 1;    
    % check stop condition
    if iters > maxIter 
        fprintf('Max Iterations Reached\n');
        break;
    end
    if err < 0.00001
        fprintf('Error reached minimum\n');
        break;
    end
%     if abs(prevErr - err) < 0.0001
%         break;
%     end
%     prevErr = err;
end

x_ = min(tx(:, 2)):0.01:max(tx(:, 2));
x_ = x_';

x_ = [ones(size(x_, 1),1) x_];
yi_ = [];
g_ = [];
w_ = [];
for i_ = 1:size(x_, 1)
    
    xt = x_(i_, :);
    % calculate intermediate values
    g = (exp(m*xt'))';
    
    g = g ./ sum(g);
    w = zeros(outputCount, expertCount);
    
    for j=1:outputCount
        w(j,:) = (v(:,:,j)*xt')';
        
    end
    
    w_ = [w_; w];
    % calculate output
    
    
    g_ = [g_; g];
    
    % output for each output dimension
    yi = (w*g')';
    yi_ = [yi_ yi];
    
end

size(w_)
figure()
plot(x_(:, 2)', yi_, '-');
hold on;
plot(tx(:, 2), tr, '+');


figure()
plot(x_(:, 2)', yi_, '-');
hold on;
plot(tx(:, 2), tr, '+');
hold on;
for i_ = 1:expertCount
    plot(x_(:, 2)', w_(:, i_), '- ');
end

size(x_)
size(g_)
figure()
plot(x_(:, 2)', yi_, '-');
hold on;
plot(tx(:, 2), tr, '+');
hold on;
for i_ = 1:expertCount
    plot(x_(:, 2)', g_(:, i_), '- ');
end

end