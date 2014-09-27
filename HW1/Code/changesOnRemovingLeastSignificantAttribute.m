function [] = changesOnRemovingLeastSignificantAttribute(X, trainingFraction, lambda )
[ numInstances numAttributes ] = size(X);
trainingMSEBefore=[];
trainingMSEAfter=[];
testMSEBefore=[];
testMSEAfter=[];
for iteration=1:100
    Y = X(randperm(end),:);
    trainingSet = Y(1:int32(numInstances*(trainingFraction)),:);
    testSet = Y(int32(numInstances*(trainingFraction))+1:end,:);
    t = size(trainingSet);
    for i=2:t(2)-1
        temp = trainingSet(:,i);
        Mean = mean(temp);
        Std = std(temp);
        trainingSet(:,i)= (trainingSet(:,i)-Mean)/Std;
        testSet(:,i) = ( testSet(:,i)-Mean)/Std;
    end
    Weights = mylinridgereg(trainingSet(:,1:t(2)-1),trainingSet(:,t(2)),lambda);
    trainingTargetValues = mylinridgeregval(trainingSet(:,1:t(2)-1), Weights);
    trainingMSE = meansquarederr(trainingSet(:,t(2)),trainingTargetValues);
    testTargetValues = mylinridgeregval(testSet(:,1:t(2)-1), Weights);
    testMSE = meansquarederr(testSet(:,t(2)),testTargetValues);
    Weights = removeLeastSignificantAttribute(Weights);
    Weights = removeLeastSignificantAttribute(Weights);
    
    newTrainingTargetValues = mylinridgeregval(trainingSet(:,1:t(2)-1), Weights);
    newTrainingMSE = meansquarederr(trainingSet(:,t(2)),trainingTargetValues);
    newTestTargetValues = mylinridgeregval(testSet(:,1:t(2)-1), Weights);
    newTestMSE = meansquarederr(testSet(:,t(2)),testTargetValues);
    
    trainingMSEBefore=[trainingMSEBefore trainingMSE];
    trainingMSEAfter=[trainingMSEAfter newTrainingMSE];
    testMSEBefore=[testMSEBefore testMSE];
    testMSEAfter=[testMSEAfter newTestMSE];
end
figure();
subplot(1,2,1);
plot(trainingMSEBefore,trainingMSEAfter,'o');
title(' Training MSE before and after removing 2 least signinficant attributes' );
xlabel(' Training MSE before removing least signinficant attributes')
ylabel(' Training MSE after removing least signinficant attributes ');
hold on;
subplot(1,2,2);
plot(testMSEBefore, testMSEAfter,'o');
title(' Test MSE before and after removing 2 least signinficant attributes' );
xlabel(' Test MSE before removing least signinficant attributes')
ylabel(' Test MSE after removing least signinficant attributes ');
hold on;
