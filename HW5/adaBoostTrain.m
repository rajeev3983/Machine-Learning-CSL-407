function [svmModels,alpha,signReverse,Accuracy] = adaBoostTrain(X,Y,maxItr)
N = size(X,1);
D = ones(1,N)/N;
signReverse = zeros(maxItr,1);
alpha = [];
svmModels = [];
X1 = [];
X2 = [];
for i=1:size(X,1)
    if( Y(i)==1 )
        X1= [ X1;X(i,:) ];
    else
        X2 = [X2;X(i,:)];
    end
end
%scatter(X1(:,1),X1(:,2),'r');
%hold on;
%scatter(X2(:,1),X2(:,2),'g');
%hold on;
predicted = zeros(N,1);
Accuracy = [];
for i=1:maxItr
    %[ trX, trY ] = getWeightedSample(X,Y,D(i,:));
    svmModel = svmtrain(X,Y,'boxconstraint',N*D(i,:)','kernel_function','linear','showplot',false,'method','SMO');
    %hold on;
    predictedY = svmclassify(svmModel,X,'Showplot',false);
    %hold on;
    et = sum(D(i,:).*(predictedY~=Y)');
    if( et== 1 )
        break;
    end
    if( et> 0.5 )
        signReverse(i,1) = -1;
        et = 1- et;
    else
        signReverse(i,1) = 1;
    end
    svmModels = [ svmModels; svmModel ];
    alpha = [ alpha; 0.5*log((1-et)/(et)) ];
    temp = [];
    for j=1:N
        temp = [ temp, D(i,j)*exp(-alpha(i)*predictedY(j)*Y(j))*signReverse(i,1) ];
    end
    temp = temp/(sum(temp));
    D = [D;temp];
    predicted = predicted + alpha(i)*predictedY*signReverse(i);
    t1 = sign(predicted);
    temp = sum(t1~=Y)/size(Y,1);
    Accuracy = [ Accuracy; temp];
end
end