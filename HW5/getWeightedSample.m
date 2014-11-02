function [trX, trY] = getWeightedSample(X,Y,D)
trX = [];
trY = [];
N = size(D,2);
Dist = zeros(N,1);
added = zeros(N,1);
Dist(1) = D(1);
for i=2:N
    Dist(i) = Dist(i-1)+D(i);
end
for i=1:N
    r = rand();
    for j=1:N
        if(Dist(j)>= r && added(j)==0 )
            trX= [trX;X(j,:)];
            trY = [trY;Y(j,:)];
            added(j)=1;
            break;
        end
    end
end
end