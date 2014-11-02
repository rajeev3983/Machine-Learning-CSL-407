function [ Ans ] = trAdaBoost(X, svmModels, Bt)
N = size(Bt,1);
start = ceil(N/2);
n = size(X,1);
temp1= ones(n,1);
temp2 = 1;
for i=start:N
    pre = svmclassify(svmModels(i,1),X);
    pre = pre==1;
    t1 = power(Bt(i),-pre);
    t2 = power(Bt(i), -0.5);
    temp1 = temp1.*t1;
    temp2 = temp2*t2;
end
Ans = temp1 >= temp2;
Ans = double(Ans);
for i= 1:size(Ans,1)
    if(Ans(i)==0)
        Ans(i)=-1;
    end
end

end