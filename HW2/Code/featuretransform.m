function Y = featuretransform( X, degree )
Size = size(X);
Y = ones(Size(1),1);
i = 2;
for j=1:degree
    k= j;
    while ( k >= 0 )
        Y = [Y, (X(:,1).^k).*(X(:,2).^(j-k))];
        i = i+ 1;
        k = k- 1;
    end
end
