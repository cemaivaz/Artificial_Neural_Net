
function [E,E_val,Ws,Vs,Ys_val] = train(limit,H,n,Ws_older,Vs_older)
  training = get_training();
  validation = get_val();
  validation = validation;
  Xs_val= validation(1,:);
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
    Ws
    Vs
    Xs_val

    [Ys_val] = validate(Ws,Vs,Xs_val);
    E_val(i) = 1/2* sum((Rs_val-Ys_val).^2);
    i = i + 11;
  end
  fprintf('Finished by the epoch : %d\n', i-1);
end