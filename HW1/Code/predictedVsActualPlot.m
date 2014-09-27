function [] = predictedVsActualPlot(bestTrainingFraction, bestLambda, X )
Y = X(randperm(end),:);
[ numInstances numAttributes ] = size(Y);
trainingSet = Y(1:int32(numInstances*(bestTrainingFraction)),:);
testSet = Y(int32(numInstances*(bestTrainingFraction))+1:end,:);
t = size(trainingSet);
for i=2:t(2)-1
    temp = trainingSet(:,i);
    Mean = mean(temp);
    Std = std(temp);
    trainingSet(:,i)= (trainingSet(:,i)-Mean)/Std;
    testSet(:,i) = ( testSet(:,i)-Mean)/Std;
end
Weights = mylinridgereg(trainingSet(:,1:t(2)-1),trainingSet(:,t(2)),bestLambda);
trainingTargetValues = mylinridgeregval(trainingSet(:,1:t(2)-1), Weights);
figure();
subplot(1,2,1);
plot(trainingTargetValues,trainingSet(:,t(2)),'o');
ylabel('Actual Value');
xlabel('Predicted Value');
title( 'Predicted Vs Actual Value for Training Set' );
hold on;
testTargetValues = mylinridgeregval(testSet(:,1:t(2)-1), Weights);
subplot(1,2,2);
plot(  testTargetValues,testSet(:,t(2)),'o');
ylabel('Actual Value');
xlabel('Predicted Value');
title( 'Predicted Vs Actual Value for Test Set' );
suptitle(strcat('Best Training Fraction = ', num2str(bestTrainingFraction),' Best Lamda = ', num2str(bestLambda) ) )
hold on;