function randOut = RandAct(upperBound)
%This function selects an action randomly between 1, and the count of all
%possible actions
randOut = ceil(rand * upperBound);
if randOut == 0
    randOut = 1;
end