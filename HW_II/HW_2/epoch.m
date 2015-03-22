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
