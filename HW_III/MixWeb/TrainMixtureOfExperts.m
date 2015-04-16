function [v, m, s] = TrainMixtureOfExperts(problemType, moeType, tx, tr, expertCount, maxIter, learningRate, decay)
% function [v, m] = TrainMixtureOfExperts(tx, tr, expertCount)
% Trains a mixture of experts logistic discriminator with given number of
% experts
% problemType: regression or classification
% moeType: cooperative or competitive
% tx: training data, samples in rows
% tr: training class labels given as membership vectors or output values in
% each row for regression problems
% expertCount: number of experts (logistic discriminators) to use


Hi_ = expertCount;
if strcmpi(problemType, 'regression')
    ptype = 1;
else
    ptype = 2;
end

if strcmpi(moeType, 'cooperative')
    mtype = 1;
else
    mtype = 2;
end

sampleCount = size(tx,1);
dim = size(tx,2) + 1;
outputCount = size(tr, 2);



min_ = min(tx);

max_ = max(tx);


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
while 1
    % choose next training instance randomly
    trSeq = randperm(sampleCount);
    for i=1:sampleCount
        k = trSeq(i);
        xt = tx(k,:);
        rt = tr(k,:);
        
       
        
        % calculate intermediate values
        g = (exp(m*xt'))';
        
        %Ph(j) = exp(-sum(([allPoi(i, 1)] - Mhj(j, :)) .^ 2) / (2 * sh(j) ^ 2));
        %g = (exp(-(sum(repmat(xt, size(m, 1), 1) - m, 2)) .^ 2))';
%
        %g = (exp(-((sum(repmat(xt(2), size(m, 1), 1) - m(:, 2), 2)) .^ 2) / (2 * s)))';

%         g = zeros(1, expertCount);
%         for ex = 1:expertCount
%             
%             g(ex) = (exp(-((sum(xt(2) - m(ex, 2), 2)) .^ 2) / (2 * s(ex) .^ 2)))';
%         end
        g = g ./ sum(g);
        w = zeros(outputCount, expertCount);
        for j=1:outputCount
            w(j,:) = (v(:,:,j)*xt')';
        end        
        % calculate output
        
        % output for each output dimension
        yi = (w*g')';
        
        % convert output through softmax if classification
        if ptype == 2
            ysi = exp(yi);
            ysi = ysi ./ sum(ysi);
            
            yih = exp(w);
            yih = yih ./ repmat( sum(yih, 1), outputCount, 1);
        end        
        
        % update parameters
        if ptype == 1
            fh = exp(sum( (repmat( rt', 1, expertCount )  - w) .^2, 1 ) .* -0.5) .* g;
        else
            fh = exp( sum( log(yih) .* repmat(rt', 1, expertCount), 1 ) ) .* g;
        end        
        fh = fh ./ sum(fh);
        
        % calculate delta v and m for each output unit
        for r=1:expertCount
            for j=1:outputCount
                if ptype == 1
                    if mtype == 1
                        dv = learningRate .* ( rt(1,j) - yi(1,j)) * g(1, r) * xt;
                    end
                    if mtype == 2
                        dv = learningRate .* ( rt(1,j) - w(j, r)) * fh(1, r) * xt;
                    end
                end
                if ptype == 2
                    if mtype == 1
                        dv = learningRate .* ( rt(1,j) - ysi(1,j)) * g(1, r) * xt;
                    end
                    if mtype == 2
                        dv = learningRate .* ( rt(1,j) - yih(j, r)) * fh(1, r) * xt;
                    end
                end
                v(r, :, j) = v(r, :, j) + dv;
                
            end
            % calculate delta m
            if ptype == 1
                if mtype == 1
                    dm = learningRate .* sum((rt - yi) .* (w(:, r)' - yi)) .* g(1, r) * xt;
                end
                if mtype == 2
                    dm = learningRate .* ( fh(1, r) - g(1, r) ) * xt;
                end
            end
            if ptype == 2
                if mtype == 1
                    dm = learningRate .* sum((rt - ysi) .* (w(:, r)' - yi)) .* g(1, r) * xt;
                end
                if mtype == 2
                    dm = learningRate .* ( fh(1, r) - g(1, r) ) * xt;
                end
            end
            m( r, : ) = m( r, : ) + dm;
            
%             const = -0.44;
%             sum_ = 0;
%             for t_ = 1:expertCount
%                m(t_, 1) = const + sum_;
%                sum_ = sum_ + 0.42;
%             end
           
%             tmpM = m( r, : ) + dm;
% 
%             if tmpM(end) <= min_
%                 m(r, end) = min_ + (max_ - min_) / (Hi_ * 2);
%                 
%             elseif tmpM(end) >= max_
%                 
%                 m(r, end) = max_ - (max_ - min_) / (Hi_ * 2);
%                 
%             else
%                 m(r, end) = tmpM(end);
%             end
%             m(r, 1) = tmpM(1);

%             if tmpM(1) <= min_
%                 m(r, 1) = min_ + (max_ - min_) / (Hi_ * 2);
%                 
%             elseif tmpM(1) >= max_
%                 
%                 m(r, 1) = max_ - (max_ - min_) / (Hi_ * 2);
%                 
%             else
%                 m(r, 1) = tmpM(1);
%             end

%             m(:, 1 ) = repmat(1, size(m, 1), 1);
        end        
    end
   
    learningRate = learningRate * decay;
    % calculate training set error

    err = TestMixtureOfExperts(problemType, tx(:,2:dim), tr, v, m, s);
    fprintf('Error: %f\n', err);
    errs(iters, 1) = err;
    % take average error of last five runs
    si = (((iters - 5) >= 0) * (iters - 5)) + 1;
    li = si + 4;
    iters = iters + 1;    
    % check stop condition
    if iters > maxIter 
        fprintf('Max Iterations Reached\n');
        break;
    end
    if err < 1e-6 || sum( errs(si:li,:) ) < sum( errs(si+1:li+1,:) )
        fprintf('Error reached minimum\n');
        break;
    end
end
end