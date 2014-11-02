load('dataset1.mat');
rng(7);
c = cvpartition(Y,'KFold',10);
maxItr = 500;
trainErr = [];
testErr = [];
Precision = 0;
Recall = 0;
ConM = zeros(2,2);
Accuracy = zeros(maxItr,1);
for i=1:10
    trInd = c.training(i);
    trX = [];
    trY = [];
    tsX = [];
    tsY = [];
    for j=1:size(X)
        if( trInd(j)==1 )
            trX = [ trX;X(j,:) ];
            trY = [ trY;Y(j,:) ];
        else
            tsX = [ tsX; X(j,:) ];
            tsY = [ tsY; Y(j,:) ];
        end
    end
    [ svmModels,alpha, signReverse,accuracy]= adaBoostTrain(trX,trY,maxItr);
    Accuracy = Accuracy + accuracy;
    %conM = adaBoostTest(trX,trY,svmModels,alpha,signReverse);
    conM = adaBoostTest(tsX,tsY,svmModels,alpha, signReverse);
    ConM = ConM + conM;
    precision = conM(1,1)/ sum(conM(:,1));
    recall =  conM(1,1)/sum(conM(1,:));
    Precision = Precision + precision;
    Recall = Recall + recall;
    fprintf('Precision for fold no. %d = %f \n', i, precision);
    fprintf('Recall for fold no. %d = %f \n', i, recall );
    fprintf('Confusion matrix for fold no. %d \n ', i );
    conM
    %trainErr = [ trainErr; trainingError];
    %testErr = [ testErr; testError ];
end
fprintf('Average Precision over 10 Fold = %f \n', Precision/10 );
fprintf('Average Recall over 10 Fold = %f \n', Recall/10 );
fprintf('Average Confusion Matrix \n');
ConM/10
Accuracy = Accuracy/10;
Accuracy = 1-Accuracy;
temp = 1:1:size(Accuracy,1);
plot(temp,Accuracy);
title( 'Accuracy Vs Number of Iterations ');
xlabel( 'No. of Iterations' );
ylabel( 'Accuracy');
