% Q-Learning
%HW-IV

close all;
clear all;
clc

alpha = 0.049;
%The below epsilon value decays throughout episodes
gamma = 0.9;

%The number of rows, and columns, respectively
rowCnt = 8;
colCnt = 8;

%The starting point: (1,1) - The lowest left corner
%The goal: (7, 6)
rowSt = 1;
rowGoal = 7;

%N: North, W: West, S: South, E: East
actNo = 4;

%st_: start
st_.row = rowCnt - rowSt + 1;
st_.column = 1;
goal.row = rowCnt - rowGoal + 1;
goal.column = 6;


% initialize Q with random numbers
Q = rand(rowCnt, colCnt, actNo);



act = 0;
cntEpis = 3000;

i_ = 1;
%%
%Episodes
while (i_ <= cntEpis)

    currentSt = st_;
    nextSt = st_;
    
 
    
    epsilon = 0.9;
    %e-Greedy method
    if(epsilon <= rand)
        [maxQ, act] = max(Q(currentSt.row, currentSt.column, :));
    else
        act = RandAct(actNo);
    end

    episRun = false;
    while (episRun == false)
       
        
        nextSt = PositionNext(currentSt, act, colCnt, rowCnt); %The action "act" is
        %taken, and the next state (position) is returned
        

        
        [maxQ, act_next] = max(Q(nextSt.row,nextSt.column,:)); %In whichever direction
        %the maximum value is produced, the corresponding state, and the
        %value thereof are returned
        
        
        if (DiffPos(nextSt, goal) == 0)
            episRun = true;
            %r: Reward
            %Gaussian with mean 100 and stdev 10
            r = normrnd(100, 10);
        else
            r = 0;
        end

        
        %e-Greedy
        if (rand < epsilon)
            act_next = RandAct(actNo);
        end
        
        
        
        Qcurrent = Q(currentSt.row, currentSt.column, act);
        Qnext = maxQ;

        %The next state when action is performed
        destPos = PositionNext(currentSt, act_next, colCnt, rowCnt);
        

        %Two orthogonal states
        [orthogSt1, orthogSt2] = OrthPos(currentSt, destPos, rowCnt, colCnt);

        %The below variable is used in choosing which way of three
        %directions to take
        directRand = rand;
        [orthogVal1, act_next_ort1] = max(Q(orthogSt1.row,orthogSt1.column,:));
        [orthogVal2, act_next_ort2] = max(Q(orthogSt2.row,orthogSt2.column,:));
        

        %With the greatest probability, it chooses the intended state, with
        %less probabilities, one of the orthogonal directions
%         if directRand <= 0.5
%             Q(currentSt.row, currentSt.column, act) = (Qcurrent + alpha * (r + gamma * Qnext - Qcurrent));
%             currentSt = nextSt;
%             act = act_next;
%         elseif directRand > 0.5 && directRand <= 0.75
%             Q(currentSt.row, currentSt.column, act) = (Qcurrent + alpha * (r + gamma * orthogVal1 - Qcurrent));
%             currentSt = orthogSt1;
%             act = act_next_ort1;
%         else
%             Q(currentSt.row, currentSt.column, act) = (Qcurrent + alpha * (r + gamma * orthogVal2 - Qcurrent));
%             currentSt = orthogSt2;
%             act = act_next_ort2;
%         end
Q(currentSt.row, currentSt.column, act) = (Qcurrent + alpha * (r + gamma * Qnext - Qcurrent));
currentSt = nextSt;
act = act_next;

%The epsilon value decays
epsilon = epsilon * 0.922;
    end
    
    i_ = i_ + 1;
end

%%
%Below are the corresponding variables showing the optimal actions, and the
%scores thereof
[max_, ind_] = max(Q, [], 3);
fprintf('Max_a Q(s, act):\n');
max_

fprintf('\n\nOptimal actions:\n');
%indDir = zeros(size(ind_,1), size(ind_, 2));
for i = 1:size(ind_, 1)
    for j = 1:size(ind_, 2)
        switch ind_(i, j)
            case 1
                indDir{i, j} = 'E';
            case 2
                indDir{i, j} = 'S';
            case 3
                indDir{i, j} = 'W';
            case 4
                indDir{i, j} = 'N';
            otherwise
                indDir{i, j} = '_';
        end
    end
end
%Bump into the wall? Hold.
indDir



