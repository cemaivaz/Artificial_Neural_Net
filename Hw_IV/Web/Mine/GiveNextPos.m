function nextPos = GiveNextPos(curPos, actionIndex, gridCols, gridRows)
nextPos = curPos;
switch actionIndex
   case 1 % east
       nextPos.col = curPos.col + 1;
   case 2 % south
       nextPos.row = curPos.row + 1;
   case 3 % west
       nextPos.col = curPos.col - 1;
   case 4 % north
       nextPos.row = curPos.row - 1;
   case 5 % hold
       nextPos = curPos;
   otherwise
      disp(sprintf('invalid action index: %d', actionIndex))
end
if(nextPos.col <= 0), nextPos.col = 1; end
if(nextPos.col > gridCols), nextPos.col = gridCols; end
if(nextPos.row <= 0), nextPos.row = 1; end
if(nextPos.row > gridRows), nextPos.row = gridRows; end
