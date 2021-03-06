% Q-Learning
%HW-IV

close all;
clear all;
clc

alpha = 0.05;
%The below epsilon value decays throughout episodes
epsilon = 0.1;
gamma = 0.9;

%The number of rows, and columns, respectively
gridrows = 8;
gridcols = 8;

%The starting point: (1,1) - The lowest left corner
%The goal: (7, 6)
startrow = 1;
goalrow = 7;

%N: North, W: West, S: South, E: East
actionCount = 4;

start.row = gridrows - startrow + 1;
start.col = 1;
goal.row = gridrows - goalrow + 1;
goal.col = 6;


% initialize Q with zeros
Q = rand(gridrows, gridcols, actionCount);



a = 0; % an invalid action
episodeCount = 1900;

%Episodes
for ei = 1:episodeCount,
    
    curpos = start;
    nextpos = start;
    
 
    
    epsilon = 0.9;
    %e-Greedy method
    if(rand > epsilon)
        [qmax, a] = max(Q(curpos.row,curpos.col,:));
    else
        a = IntRand(1, actionCount);
    end

    episodeFinished = 0;
    while(episodeFinished == 0)
       
        
        nextpos = GiveNextPos(curpos, a, gridcols, gridrows); %The action "a" is
        %taken, and the next state (position) is returned
        
        if(PosCmp(nextpos, goal) == 0), 
            episodeFinished = 1;
            %r: Reward
            %Gaussian with mean 100 and stdev 10
            r = normrnd(100, 10); 
        else
            r = 0;
        end

        
        [qmax, a_next] = max(Q(nextpos.row,nextpos.col,:)); %In whichever direction
        %the maximum value is produced, the corresponding state, and the
        %value thereof are returned
        
        %e-Greedy
        if(rand <= epsilon)
            a_next = IntRand(1, actionCount);
        end
        
        
        
        curQ = Q(curpos.row, curpos.col, a);
        nextQ = qmax;

        %The next state when action is performed
        destPos = GiveNextPos(curpos, a_next, gridcols, gridrows);
        

        %Two orthogonal states
        [orthogSt1, orthogSt2] = ProbPos(curpos, destPos, gridrows);

        %The below variable is used in choosing which way of three
        %directions to take
        directRand = rand;
        [orthogVal1, a_next_ort1] = max(Q(orthogSt1.row,orthogSt1.col,:));
        [orthogVal2, a_next_ort2] = max(Q(orthogSt2.row,orthogSt2.col,:));
        
        %With the greatest probability, it chooses the intended state, with
        %less probabilities, one of the orthogonal directions
        if directRand <= 0.5
            Q(curpos.row, curpos.col, a) = (curQ + alpha*(r + gamma*nextQ - curQ));
            curpos = nextpos;
            a = a_next;
        elseif directRand > 0.5 && directRand <= 0.75
            Q(curpos.row, curpos.col, a) = (curQ + alpha*(r + gamma*orthogVal1 - curQ));
            curpos = orthogSt1;
            a = a_next_ort1;
        else
            Q(curpos.row, curpos.col, a) = (curQ + alpha*(r + gamma*orthogVal2 - curQ));
            curpos = orthogSt2;
            a = a_next_ort2;
        end
        
        %The epsilon value decays
        epsilon = epsilon * 0.922;
    end
    

end


%Below are the corresponding variables showing the optimal actions, and the
%scores thereof
[max_, ind_] = max(Q, [], 3);
fprintf('Max_a Q(s, a):\n');
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



