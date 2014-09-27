function [] = lindiscriminant(X,Y)
degree = 3;
temp = size(X);
c1 = [];
c2 = [];
for i=1:temp(1)
    if( Y(i)==0 )
        c1 = [c1;X(i,:)];
    else
        c2 = [c2;X(i,:)];
    end
end
mu1 = mean(c1);
mu2 = mean(c2);
Sigma = zeros(temp(2),temp(2));
c1Size = size(c1);
c2Size = size(c2);
for i=1:c1Size(1)
    Sigma = Sigma + (c1(i,:)- mu1)'*(c1(i,:)-mu1);
end
for i=1:c2Size(1)
    Sigma = Sigma + (c2(i,:)-mu2)'*(c2(i,:)-mu2);
end
Sigma = Sigma./ (temp(1)-2);
X1 =[1:0.1:7];
Y1= [1:0.1:7];
Size = size(X1);
map = zeros(Size(2),Size(2));
for i=1:Size(2)
    for j=1:Size(2)
        X2 = [X1(i),Y1(j)];
        X2 = featuretransform(X2,degree);
        delta1 = discriminant(X2,Sigma,mu1,c1Size(1)/(c1Size(1)+c2Size(1)));
        delta2 = discriminant(X2,Sigma,mu2,c2Size(1)/(c1Size(1)+c2Size(1)));
        map(j,i)= abs(delta1-delta2);
    end
end
contour(X1,Y1,map,0.2);
hold on;
title( 'Decision boundary obtained using Discriminant Function');