function [conM] = adaBoostTest(X,Y,svmModels,alpha,signReverse)
predicted = zeros(size(X,1),1);
for i=1:size(svmModels)
    temp = svmclassify(svmModels(i),X,'Showplot',false);
    predicted = predicted + alpha(i)*temp*signReverse(i,1);
end
predicted = sign(predicted);
%error = (sum(predicted~=Y))/(size(X,1));
conM = zeros(2,2);
conM(1,1) = sum( ( (predicted==1) + (Y==1))==2 );
conM(1,2) = sum( ((predicted==-1) + (Y==1))==2);
conM(2,1 ) = sum( ((predicted==1) + (Y==-1))==2);
conM(2,2) = sum( ((predicted==-1) + (Y==-1))==2);
conM = conM/(sum(sum(conM)));
end