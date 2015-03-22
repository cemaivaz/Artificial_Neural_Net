
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