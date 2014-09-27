function [] = plotdecisionboundary(w,X,Y)
degree = 3;
X1 =[1:0.1:7];
Y1= [1:0.1:7];
Size = size(X1);
map = zeros(Size(2),Size(2));
for i=1:Size(2)
    for j=1:Size(2)
        X2 = [ X1(i), Y1(j)];
        X2 = featuretransform(X2,degree);
        map(j,i) = X2*w;
    end
end
contour(X1,Y1,map,0.1);
hold on;
title( 'Decision boundary obtained using Logistic Regression' );