function diff_ = DiffPos(position1, position2)
%This function calculated the distance between two positions
colDiff = position1.column - position2.column;
if colDiff == 0
    diff_ = position1.row - position2.row;
else
    diff_ = colDiff;
end