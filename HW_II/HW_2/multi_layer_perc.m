function [Z,Y] = multi_layer_perc(Ws,V,X,H)
  Z = ones(1,H);
  Z(1) = 1;
  for h = 2:H,
    Z(h) = sigmoid(Ws(h,:),X);
  end

  fprintf('_________________');
  V
  Z
  Y = V*Z';
end