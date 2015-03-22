clear all;
clc;
fprintf('hhhhh');
[H,K,Ws,Vs,n,Xs,Rs] = initiate(2,1,0.1,get_training);
[E,E_val,Ws,Vs,Ys_val] = train(300,2,0.1,Ws,Vs);