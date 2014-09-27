load('credit.mat')
degree = 3;
figure;
[numPoints, temp ] = size(data);
for i=1:numPoints
    if label(i) == 0
        scatter( data(i,1),data(i,2),'r' );
    else
        scatter( data(i,1),data(i,2),'b' );
    end
    hold on;
end
transformedData =  featuretransform( data, degree);
Size = size(transformedData);
options = optimset('GradObj', 'on', 'MaxIter', 100);
initial_w = zeros(Size(2),1);
[w objval] = fminunc(@(w)(objgradcompute(w, transformedData, label)), initial_w, options)
plotdecisionboundary(w,transformedData,label);
figure;
for i=1:numPoints
    if label(i) == 0
        scatter( data(i,1),data(i,2),'r' );
    else
        scatter( data(i,1),data(i,2),'b' );
    end
    hold on;
end
lindiscriminant(transformedData,label);