function nextPos = PositionNext(PosCurrent, actNo, colCnt, rowCnt)
%This function determines the next action, given the current position and
%action
nextPos = PosCurrent;
if actNo == 1
    nextPos.column = PosCurrent.column + 1;
elseif actNo == 2
    nextPos.row = PosCurrent.row + 1;
elseif actNo == 3
    nextPos.column = PosCurrent.column - 1;
elseif actNo == 4
    nextPos.row = PosCurrent.row - 1;
end

%% 
%If the boundary limit is exceeded, the score is assigned the maximum, or
%minimum value
if(nextPos.row > rowCnt)
    nextPos.row = rowCnt; 
end
if(nextPos.row < 1)
    nextPos.row = 1; 
end
if(nextPos.column < 1)
    nextPos.column = 1; 
end
if(nextPos.column > colCnt)
    nextPos.column = colCnt;
end