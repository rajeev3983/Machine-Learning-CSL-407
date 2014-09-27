function Wdash = removeLeastSignificantAttribute(W)
minValue = 10000000;
leastSignificant = 1;
for i=1:length(W)
    if( W(i)~=0 && abs(W(i))< minValue )
        minValue = abs(W(i));
        leastSignificant = i;
    end
end
Wdash = W;
Wdash(leastSignificant) = 0;
end