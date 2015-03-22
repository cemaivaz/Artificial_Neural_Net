function [Ws,Vs,Ys,Zs,d_v,d_w] = epoch(Ws,Vs,Xs,Rs,n,H)
  T = length(Xs);
  K = 1;
  D = length(Xs(1,:));
  d_v = zeros(H,K);
  d_w = zeros(H,D);
  Zs = ones(T,H);
  for t =  randperm(T),
    [Z,Y] = multi_layer_perc(Ws,Vs,Xs(t,:),H);
    Ys(t,:) = Y;
    %Zs(t,:) = Z;
    for h = 1:H,
      back_err = 0;
      for i = 1:K,
    d_v(h,:) = d_v(h, :) + sum(Z(h)*(Rs(t)-Ys(t,i)));
    back_err = back_err + (Rs(t)-Ys(t,i))*Vs(i,h);
      end
      %fprintf('value of t: %d, h:%d, \n', t,h);
      d_w(h,:) = d_w(h, :) + back_err*Z(h)*(1-Z(h))*Xs(t,:);
    end
  end

  for h = 1:H,
    Ws(h,:) = Ws(h, :) + n*d_w(h,:);
  end
  Vs = Vs + n*d_v';
end

function [Z,Y] = multi_layer_perc(Ws,V,X,H)
  Z = ones(1,H);
  Z(1) = 1;
  for h = 2:H,
    Z(h) = sigmoid(Ws(h,:),X);
  end
  Y = V*Z';
end

function [Ys] = validate(Ws,Vs,Xs)
  T = length(Xs);
  H = length(Ws);
  for t =  randperm(T),
    [Z,Y] = multi_layer_perc(Ws,Vs,Xs(t,:),H);
    Ys(t,:) = Y;
  end
end

function [res] = sigmoid(Ws, X)
  sum = 0;
  for i = 1:length(Ws)
    sum = sum + Ws(i) * X(i); 
  end
  res = 1 / (1 + exp(-sum));
end

function [E,E_val,Ws,Vs,Ys_val] = train(limit,H,n,Ws_older,Vs_older)
  training = get_training();
  validation = get_val();
  validation = validation';
  Xs_val= Xs_create(validation(1,:));
  Rs_val = validation(2,:)';
  [H,K,Ws,Vs,n,Xs,Rs] = initiate(H,1,n,training);
  E = ones(1,limit+1);
  if length(Ws_older) > 0
     Ws = Ws_older;
     Vs = Vs_older;
  end
  i = 2;
  E(1) = 10000;
  E_Val = E;

  while ( (E(i) < E(i-1))  || (i < 110)) && i < limit+1,
    [Ws,Vs,Ys,Z] = epoch(Ws,Vs,Xs,Rs,n,H);
    E(i) = 1/2* sum((Rs-Ys).^2);
    [Ys_val] = validate(Ws,Vs,Xs_val);
    E_val(i) = 1/2* sum((Rs_val-Ys_val).^2);
    i = i + 11;
  end
  fprintf('Finished by the epoch : %d\n', i-1);
end

function [tr] = get_training()


%Precision is chosen to be large
format long

fprintf('Model is being trained..\n\n');


%Hidden unit numbers are shown below
Hidd = [4];

%Training data
dataTr = textread('training.txt', '%s');

%The x values thereof
xt = dataTr(1:2:length(dataTr) - 1);

%The y values thereof
rt = dataTr(2:2:length(dataTr));

%Data are shuffled
randOrd = randperm(length(xt));

tmpXt = [];
tmprt = [];
for i = 1:length(xt)
    tmpXt = [tmpXt; str2double(xt(i))];
    tmprt = [tmprt; str2double(rt(i))];
    
end



xt = tmpXt;
rt = tmprt;

xt = xt(randOrd);

rt = rt(randOrd);

tr = [xt rt];
end

function [val] = get_val()
%Hidden unit numbers are shown below
Hidd = [4];

%Training data
dataTr = textread('validation.txt', '%s');

%The x values thereof
xt = dataTr(1:2:length(dataTr) - 1);

%The y values thereof
rt = dataTr(2:2:length(dataTr));

%Data are shuffled
randOrd = randperm(length(xt));

tmpXt = [];
tmprt = [];
for i = 1:length(xt)
    tmpXt = [tmpXt; str2double(xt(i))];
    tmprt = [tmprt; str2double(rt(i))];
    
end



xt = tmpXt;
rt = tmprt;

xt = xt(randOrd);

rt = rt(randOrd);

tr = [xt rt];
end
%  [H,K,Ws,Vs,n,Xs,Rs] = initiate(H,1,n,training);
function [H,K,Ws,Vs,n,Xs,Rs] = initiate(H,K,n,training)

 K = K;
 H = H;
 n = n;
 Xs = training(:, 1);
 Rs = training(:, 2);
     %First, the matrix Wij is filled with numbers randomly produced
    for h = 0:H
        for j = 0:dimenNo
            Ws(h + 1, j + 1) = rand() * 0.02 - 0.01;
        end
    end
    
    %The matrix Vih is being filled with those numbers randomly produced
    for i = 1:K
        for h = 0:H
            Vs(i, h + 1) = rand() * 0.02 - 0.01;
        end
    end
 
end
