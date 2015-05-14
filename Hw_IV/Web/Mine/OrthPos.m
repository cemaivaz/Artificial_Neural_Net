function [posOut1, posOut2] = OrthPos(position1, position2, colCnt, rowCnt)
%CHECKPOS Summary of this function goes here
%   Detailed explanation goes here

%This function determines two orthogonal positions
rowDiff = abs(position1.row - position2.row);

posOut1 = position1;
posOut2 = position1;

if rowDiff == 0
    posOut1.row = position1.row + 1;
    posOut2.row = position1.row - 1;
    
else
    posOut1.column = position1.column + 1;
    posOut2.column = position1.column - 1;
end

%% 
%If the boundary limit is exceeded, the score is assigned the maximum, or
%minimum value

if(posOut1.column < 1)
    posOut1.column = 1; 
end
if(posOut1.column > colCnt)
    posOut1.column = colCnt; 
end
if(posOut1.row < 1)
    posOut1.row = 1; 
end
if(posOut1.row > rowCnt)
    posOut1.row = rowCnt; 
end
if(posOut2.column <1)
    posOut2.column = 1; 
end
if(posOut2.column > colCnt)
    posOut2.column = colCnt;
end
if(posOut2.row < 1)
    posOut2.row = 1; 
end
if(posOut2.row > rowCnt)
    posOut2.row = rowCnt; 
end

