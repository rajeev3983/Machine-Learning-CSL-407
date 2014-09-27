function [ objval gradval ] = objgradcompute(w, X, Y)
T = sigmoid(X*w);
lambda = 0.1;
temp = size(X);
objval = 0;
gradval = zeros(temp(2),1);
for i=1:temp(1)
    objval = objval - (Y(i)*log(T(i)) + (1-Y(i))*log(1-T(i)));
end
objval = objval + lambda*sum(w.^2)/2;
objval = objval/(temp(1));
for j=1:temp(2)
    for i=1:temp(1)
        gradval(j) = gradval(j) + ((T(i)-Y(i))*X(i,j));
    end
    gradval(j) = gradval(j) + lambda*w(j);
    gradval(j)= gradval(j)/temp(1);
end
end