clear all;
clc;
%[H,K,Ws,Vs,n,Xs,Rs] = initiate(2,1,0.1,get_training);
%[E,E_val,Ws,Vs,Ys_val] = train(300,2,0.1,Ws,Vs);


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

%xt = [0 0; 0 1; 1 0; 1 1];
%rt = [0;1;1;0];

size_ = size(xt, 2);



%{
ws1 = N.weights{1};
ws2 = N.weights{2};
x_points = linspace(min(xt), max(xt),100)';

%x_points = linspace(0, 1, 100)';
res = [x_points repmat(1, size(x_points, 1), 1)];

res = ws1 * res';
tmpRes = exp(-res(1:end-1,:));
res2 = 2./(1+tmpRes)-1;
res2 = [res2; repmat(1,1,size(x_points, 1))];

layer_2nd = ws2 * res2;

layer_2nd = 2./(1 + exp(-layer_2nd)) - 1;

res = layer_2nd;
figure(1);
plot(xt, rt, '+')
hold on;
plot(x_points, res, '-');
hold off;

%}


hidd = [2 4 8];
cnt = 1;
for i = 1:length(hidd)
    N = backprop([size_ hidd(i) 1],0.1,0.5,0.00033,xt,rt);
    fprintf('Hidden unit #: %d, mse: %0.22f, last mse: %0.22f\n\n', hidd(i), mean(N.error), N.mse);
    str = strcat('Hidden unit #: ', int2str(hidd(i)));
    figure(cnt)

    plot((1:length(N.error)), N.error, '-');
        title(str)
    xlabel('Epochs')
    ylabel('MSE')
    hold off;
    
    
    cnt = cnt + 1;
    
    
    
    
    
    ws1 = N.weights{1};
    ws2 = N.weights{2};
    x_points = linspace(min(xt), max(xt),100)';
    
    %x_points = linspace(0, 1, 100)';
    x_vals = [x_points repmat(1, size(x_points, 1), 1)];
    
    resMod = ws1 * x_vals';
    tmpRes = exp(-resMod(1:end-1,:));
    res2 = 2./(1+tmpRes)-1;
    res2Mod = [res2; repmat(1,1,size(x_points, 1))];
    
    layer_2nd = ws2 * res2Mod;
    
   output = 2./(1 + exp(-layer_2nd)) - 1;
    
    res = output;
    figure(cnt);
    plot(xt, rt, '+')
    hold on;
    plot(x_points, res, '-');
    
    %first layer weights
    ws = N.weights{1};
    hold on;
    y_vals = ws * x_vals';
    plot(x_points, y_vals)
            title(str)

    hold off;
    cnt = cnt + 1;
    
    figure(cnt)
    for q = 1:size(res2, 1)
        
        plot(x_points, res2(q, :), '-');
        hold on;
        plot(xt, rt, '+')
        hold on;
        plot(x_points, res, '-');
        
        hold on;
        
    end
    cnt = cnt + 1;
            title(str)

    hold off;
    
    figure(cnt)
    Sec_Layer_Curves = repmat(ws2', 1, size(res2Mod, 2)) .* res2Mod;
    for q = 1:size(Sec_Layer_Curves, 1)
        
        curve_2nd_weight = Sec_Layer_Curves(q, :);
        plot(x_points, curve_2nd_weight, '-');
        hold on;
        plot(xt, rt, '+')
        hold on;
        plot(x_points, res, '-');
        
        hold on;
    end
            title(str)

    cnt = cnt + 1;
%     hold off;
    
end
