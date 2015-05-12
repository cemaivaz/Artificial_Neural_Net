function c = PosCmp(pos1, pos2)
c = pos1.row - pos2.row;
if(c == 0)
    c = c + pos1.col - pos2.col;
end