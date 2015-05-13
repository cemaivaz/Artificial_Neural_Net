
% CliffWalkingQLearning: implements the cliff-walking problem using the
% Q-Learning method

% You can pass the parameters of the problem, through the options
% structure, otherwise a default settings is used for running the program.

% written by: Sina Iravanian - June 2009
% sina@sinairv.com
% Please send your comments or bug reports to the above email address.

close all;
clear all;
clc

gamma = 0.9;
alpha = 0.05;
epsilon = 0.1;
gridcols = 8;
gridrows = 8;
fontsize = 16;
showTitle = 1;

episodeCount = 2000;
selectedEpisodes = [20 200 700 1000 2000];
isKing = 0;
canHold = 0;

start.row = 1;
start.col = 1;
goal.row = 7;
goal.col = 6;


selectedEpIndex = 1;
if(isKing ~= 0),  actionCount = 8; else actionCount = 4; end
if(canHold ~= 0 && isKing ~= 0), actionCount = actionCount + 1; end

% initialize Q with zeros
Q = rand(gridrows, gridcols, actionCount);

a = 0; % an invalid action
% loop through episodes
for ei = 1:episodeCount,
    %disp(sprintf('Running episode %d', ei));
    curpos = start;
    nextpos = start;
    
 
    
    %epsilon or greedy
    if(rand > epsilon) % greedy
        [qmax, a] = max(Q(curpos.row,curpos.col,:));
    else
        a = IntRand(1, actionCount);
    end

    %epsilon = epsilon * 0.975;
    episodeFinished = 0;
    while(episodeFinished == 0)
        % take action a, observe r, and nextpos
        nextpos = GiveNextPos(curpos, a, gridcols, gridrows);
        if(PosCmp(nextpos, goal) == 0), 
            episodeFinished = 1;
            r = 100; 
%         elseif(nextpos.row == gridrows && 1 < nextpos.col && nextpos.col < gridcols) 
%             episodeFinished = 1;
%             r = 0; 
        else
            r = 0;
        end

        % choose a_next from nextpos
        [qmax, a_next] = max(Q(nextpos.row,nextpos.col,:));
        if(rand <= epsilon) % greedy
            a_next = IntRand(1, actionCount);
        end
        
        % update Q: Sarsa
        curQ = Q(curpos.row, curpos.col, a);
        nextQ = qmax; %Q(nextpos.row, nextpos.col, a_next);
%         Q(curpos.row, curpos.col, a) = curQ + alpha*(r + gamma*nextQ - curQ);
        
        
        destPos = GiveNextPos(curpos, a_next, gridcols, gridrows);
        

        [orthogSt1, orthogSt2] = ProbPos(curpos, destPos, gridrows);
        
        fprintf('___');
        curpos
        a_next
        destPos
        fprintf('//////');
        orthogSt1
        
        orthogSt2
        
        orthogVal1 = max(Q(orthogSt1.row,orthogSt1.col,:));
        orthogVal2 = max(Q(orthogSt2.row,orthogSt2.col,:));
        
        Q(curpos.row, curpos.col, a) = .5 * (curQ + alpha*(r + gamma*nextQ - curQ));
        Q(curpos.row, curpos.col, a) = Q(curpos.row, curpos.col, a) + .25 * (curQ + alpha*(r + gamma*orthogVal1 - curQ));
        Q(curpos.row, curpos.col, a) = Q(curpos.row, curpos.col, a) + .25 * (curQ + alpha*(r + gamma*orthogVal2 - curQ));
        
        
        curpos = nextpos; a = a_next;
    end % states in each episode
    
    % if the current state of the world is going to be drawn ...
%     if(selectedEpIndex <= length(selectedEpisodes) && ei == selectedEpisodes(selectedEpIndex))
%         curpos = start;
%         rows = []; cols = []; acts = [];
%         for i = 1:(gridrows + gridcols) * 10,
%             [qmax, a] = max(Q(curpos.row,curpos.col,:));
%             nextpos = GiveNextPos(curpos, a, gridcols, gridrows);
%             rows = [rows curpos.row];
%             cols = [cols curpos.col];
%             acts = [acts a];
% 
%             if(PosCmp(nextpos, goal) == 0), break; end
%             curpos = nextpos;
%         end % states in each episode
%         
%         %figure;
%         figure('Name',sprintf('Episode: %d', ei), 'NumberTitle','off');
%         DrawCliffEpisodeState(rows, cols, acts, start.row, start.col, goal.row, goal.col, gridrows, gridcols, fontsize);
%         if(showTitle == 1),
%             title(sprintf('Cliff Walking Q-Learning - episode %d - (\\epsilon: %3.3f), (\\alpha = %3.4f), (\\gamma = %1.1f)', ei, epsilon, alpha, gamma));
%         end
%         
%         selectedEpIndex = selectedEpIndex + 1;
%     end

end % episodes loop





