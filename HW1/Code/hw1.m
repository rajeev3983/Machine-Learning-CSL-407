load('abalone.mat')
[ numInstances numAttributes ] = size(abalone);
Gender = [];
for i = 1:numInstances
    if abalone(i,1)==0
        Gender = [Gender;1 0 0];
    elseif abalone(i,1)==1
        Gender = [Gender;0 1 0];
    else
        Gender = [Gender; 0 0 1];
    end
end
X = [ones(numInstances,1) Gender abalone(:,2:numAttributes)];
numIterations = 100;
figure();
minAverageMSE=[];
minAverageMSELambda=[];
fractionValues = [];
for fraction=0.1:0.1:0.9
    averageTrainingMSE = [];
    averageTestMSE = [];
    Lambdas= [];
    minMSE = 100000;
    minLambda = 0;
    for lambda = 0:0.05:1
        cummulativeTrainingMSE = 0;
        cummulativeTestMSE = 0;
        for iterations=1:numIterations
            Y = X(randperm(end),:);          
            trainingSet = Y(1:int32(numInstances*(fraction)),:);
            testSet = Y(int32(numInstances*(fraction))+1:end,:);
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
            cummulativeTrainingMSE = cummulativeTrainingMSE + trainingMSE;
            cummulativeTestMSE = cummulativeTestMSE + testMSE;
        end
        averageTrainingMSE = [ averageTrainingMSE cummulativeTrainingMSE/numIterations];
        averageTestMSE = [ averageTestMSE cummulativeTestMSE/numIterations];
        Lambdas = [Lambdas lambda];
        if( cummulativeTestMSE/numIterations < minMSE )
            minMSE = cummulativeTestMSE/numIterations;
            minLambda =  lambda;
        end
    end
    subplot(3,3, 10*fraction );
    plot(Lambdas,averageTrainingMSE, 'b');
    title(strcat('Training Fraction =',num2str(fraction)));
    xlabel('lambda');
    ylabel('Average Mean Square Error');
    axis([0 1 4.5 5.5])
    hold on;
    plot(Lambdas,averageTestMSE, 'r');
    legend('Average Training MSE','Average Test MSE' );
    hold on;
    minAverageMSE = [ minAverageMSE, minMSE ];
    minAverageMSELambda = [ minAverageMSELambda, minLambda ];
    fractionValues = [ fractionValues fraction ]; 
end
figure();
subplot(1,2,1);
plot(fractionValues, minAverageMSE );
hold on;
xlabel('Training Fraction');
ylabel('Minimum Average MSE ');
title(' Minimum Average MSE for Different Training Fraction '); 
subplot(1,2,2);
plot(fractionValues, minAverageMSELambda );
hold on;
xlabel('Training Fraction');
ylabel('Lambda for Minimum Average MSE ');
title(' Value of Lambda for Minimum Average MSE for different Training Fraction ' );
bestFraction = 0.1;
bestLambda = 0;
minMSE = 1000;
for i=1:length(minAverageMSE)
    if ( minAverageMSE(i) < minMSE )
        minMSE = minAverageMSE(i);
        bestFraction = fractionValues(i);
        minLambda = minAverageMSELambda(i);
    end
end
predictedVsActualPlot( bestFraction,minLambda ,X);
changesOnRemovingLeastSignificantAttribute(X,bestFraction,minLambda);