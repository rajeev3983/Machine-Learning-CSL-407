function W = mylinridgereg( X, T, lambda )
t = size(X);
W = pinv((X')*X + lambda*eye(t(2)))* (X'*T);
end