function [trX, trY] = getWeightedSample2(X,Y,D)
trX = [];
trY = [];
N = size(D,2);
for i=1:40
    [mx, j ]= max(D);
    trX = [ trX; X(j,:) ];
    trY = [ trY; Y(j,:) ];
    D(1,j) = 0;
end
end