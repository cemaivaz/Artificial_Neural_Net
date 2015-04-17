

clear all;
close all;
clc

format long

%Expert numbers are shown below
NH = [2; 3; 5];

%The number of epochs is shown below
thresh = 800;

fprintf('Model is being trained..\n\n');



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

tx2 = xt(randOrd);

tr2 = rt(randOrd);



%Validation data are being read

dataVal = textread('validation.txt', '%s');

xtVal = dataVal(1:2:length(dataVal) - 1);

rtVal = dataVal(2:2:length(dataVal));

%randOrd = randperm(length(xtVal));


tmpXtVal = [];
tmprtVal = [];
for i = 1:length(xtVal)
    tmpXtVal = [tmpXtVal; str2double(xtVal(i))];
    tmprtVal = [tmprtVal; str2double(rtVal(i))];
    
end

xtVal = tmpXtVal;
rtVal = tmprtVal;



vx2 = xtVal(randOrd);
vr2 = rtVal(randOrd);


for i_ = 1:length(NH)
    [v, m, s] = trainMOE(tx2, tr2, NH(i_), 1000, 0.85, 0.99);
    [err, cr] = testMOE(vx2, vr2, v, m, s);

    
end

fprintf('\nValidation error: %.26f\n', err);