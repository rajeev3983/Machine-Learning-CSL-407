numTrainingSamples = 10;
c1TrainingData = normrnd(2,0.2,[1 numTrainingSamples]);
c1Mean = mean(c1TrainingData);
c1Sigma = std(c1TrainingData);
c2TrainingData = normrnd(3,0.2,[1 numTrainingSamples]);
c2Mean = mean(c2TrainingData);
c2Sigma = std(c2TrainingData);
c3TrainingData = normrnd(4,0.2,[1 numTrainingSamples]);
c3Mean = mean(c3TrainingData);
c3Sigma = std(c3TrainingData);
Sigma = ( c1Sigma^2 + c2Sigma^2 + c3Sigma^2 )/27;
Y = zeros(1,numTrainingSamples);
scatter(c1TrainingData, Y,'o' );
hold on;
scatter(c2TrainingData, Y,'o' );
hold on;
scatter(c3TrainingData, Y,'o' );
legend('Class 1 training Points','Class 2 training Points','Class 3 training points');
hold on;
figure;
testData = 1:0.1:5;
c1pdf = normpdf(testData,c1Mean,c1Sigma);
c2pdf = normpdf(testData,c2Mean,c2Sigma);
c3pdf = normpdf(testData,c3Mean,c3Sigma);
plot(testData,c1pdf,'b');
hold on;

plot(testData,c2pdf,'g');
hold on;

plot(testData,c3pdf,'r');
hold on;
plot(testData,c1pdf/3,':b');
hold on;
plot(testData,c2pdf/3,':g');
hold on;
plot(testData,c3pdf/3,':r');
hold on;
legend('Class 1 likelihood','Class 2 likelihood', 'Class 3 likelihood', 'Class 1 Posterior', 'Class 2 Posterior','Class 3 Posterior');

c1Posterior = testData*c1Mean/(Sigma) - ( 0.5*(c1Mean^2)/(Sigma)) + log(1/3);
c2Posterior = testData*c2Mean/(Sigma) - ( 0.5*(c2Mean^2)/(Sigma)) + log(1/3);
c3Posterior = testData*c3Mean/(Sigma) - ( 0.5*(c3Mean^2)/(Sigma)) + log(1/3);
figure;
scatter(c1TrainingData, Y,'o' );
hold on;
scatter(c2TrainingData, Y,'o' );
hold on;
scatter(c3TrainingData, Y,'o' );
hold on;
plot(testData,c1Posterior,'b');
hold on;
plot(testData,c2Posterior,'g');
hold on;
plot(testData,c3Posterior,'r');
hold on;
legend('Class 1 Training Data', 'Class 2 Training Data', 'Class 3 Training Data', 'Class 1 Discriminant', 'Class 2 Discriminant', 'Class 3 Discriminant' );
temp = size(c1Posterior);
result = zeros(1,temp(2));
for i=1:temp(2)
    if ( c1Posterior(i) >= c2Posterior(i) && c1Posterior(i) >= c3Posterior(i))
        result(i)=1;
    elseif ( c2Posterior(i) >= c1Posterior(i) && c2Posterior(i) >= c3Posterior(i) )
        result(i)=2;
    else
        result(i)=3;
    end
end
figure;
plot(testData,result,'*');
xlabel('Test Data');
ylabel('Predicted Class');
title('Class Predicted by Discriminant');