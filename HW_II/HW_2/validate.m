
function [Ys] = validate(Ws,Vs,Xs)
  T = length(Xs);
  H = length(Ws);
  
  for t =  randperm(T),
    [Z,Y] = multi_layer_perc(Ws,Vs,Xs(t,:),H);
    Ys(t,:) = Y;
  end
end