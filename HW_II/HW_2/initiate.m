function [H,K,Ws,Vs,n,Xs,Rs] = initiate(H,K,n,training)

 K = K;
 H = H;
 n = n;
 dimenNo = 1;
 Xs = training(:, 1);
 Rs = training(:, 2);
     %First, the matrix Wij is filled with numbers randomly produced
    for h = 1:H
        for j = 0:dimenNo
            Ws(h + 1, j + 1) = rand() * 0.02 - 0.01;
        end
    end
    
    %The matrix Vih is being filled with those numbers randomly produced
    for i = 1:K
        for h = 1:H
            Vs(i, h) = rand() * 0.02 - 0.01;
        end
    end
 
end
