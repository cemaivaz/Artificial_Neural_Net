
function [res] = sigmoid(Ws, X)
  sum = 0;
  Ws
  X
  for i = 2:length(Ws)
    sum = sum + Ws(i) * X(i - 1); 
  end
  
  sum = sum + Ws(1);
  
  res = 1 / (1 + exp(-sum));
end