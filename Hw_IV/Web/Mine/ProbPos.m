function [posOut1, posOut2] = ProbPos(pos1, pos2, no)
%CHECKPOS Summary of this function goes here
%   Detailed explanation goes here

rowDiff = abs(pos1.row - pos2.row);

posOut1 = pos1;
posOut2 = pos1;
colDiff = 0;
if rowDiff == 0
    colDiff = abs(pos1.col - pos2.col);
end

if (pos1.row == no || pos1.row == 1) && (pos1.col ~= 1 && pos1.col ~= no)
    
    if rowDiff == 0
        addVal = 0;
        if pos1.row == no
            addVal = -1;
        else
            addVal = 1;
        end
        posOut1.row = pos1.row + addVal;
        
        
    end
    
elseif pos1.col == 1 || pos1.col == no
    if colDiff == 0
        if pos1.row == 1
            if pos1.col == 1
                posOut1.col = 1 + (1 - colDiff);
                posOut1.row = 1 + (1 - rowDiff);
            else
                posOut1.col = no - (1 - colDiff);
                posOut1.row = 1 + (1 - rowDiff);
            end
        elseif pos1.row == no
            if pos1.col == 1
                posOut1.col = 1 + (1 - colDiff);
                posOut1.row = no - (1 - rowDiff);
            else
                posOut1.col = no - (1 - colDiff);
                posOut1.row = no - (1 - rowDiff);
            end
        else
            addVal = 0;
            if pos1.col == no
                addVal = -1;
            else
                addVal = 1;
            end
            posOut1.col = pos1.col + addVal;
        end
        
        
    end    
else 
    if rowDiff ~= 0
        posOut1.col = pos1.col - 1;
        posOut2.col = pos1.col + 1;
        
    else
        posOut1.row = pos1.row - 1;
        posOut2.row = pos1.row + 1;
    end

end

