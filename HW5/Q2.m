load('recvstalkmini.mat');
perc = 10;
rng(1);
k = 100/perc;
c = cvpartition(trlabel,'KFold',k);
c2 = cvpartition(tslabel,'KFold',k);
Error = [];
for fold=1:10
    trInd = c.training(mod(fold,k)+1);
    tsInd = c.training(mod(fold,k)+1);
    trtrainX = [];
    trtrainY = [];
    trtestX = [];
    trtestY = [];

    tstrainX = [];
    tstrainY = [];
    tstestY = [];
    tstestX = [];

    for j=1:size(trdata,1)
         if( trInd(j)==1 )
             trtrainX = [ trtrainX;trdata(j,:) ];
             trtrainY = [ trtrainY;trlabel(j,:) ];
         else
             trtestX = [ trtestX; trdata(j,:) ];
             trtestY = [ trtestY; trlabel(j,:) ];
         end
    end

    for j=1:size(tsdata,1)
            if( tsInd(j)==1 )
                tstrainX = [ tstrainX; tsdata(j,:) ];
                tstrainY = [ tstrainY; tslabel(j,:) ];
            else
                tstestX = [ tstestX; tsdata(j,:) ];
                tstestY = [ tstestY; tslabel(j,:) ];
            end
    end

    Trdata = trtestX;
    Trlabel = trtestY;
    Tsdata = tstestX;
    Tslabel = tstestY;

    n = size(Trdata,1);
    m = size(Tsdata,1);
    N= 1;
    w = ones(n+m,1)/(n+m);
    svmModels = [];
    Bt = [];

    for i=1:N
        pt = w/(sum(w));
        svmModel = svmtrain( [Trdata;Tsdata],[Trlabel;Tslabel],'boxconstraint',(n+m)*pt);
        svmModels = [ svmModels; svmModel ];
        predictedts = svmclassify(svmModel,Tsdata);
        et =  sum( ( w(n+1:end,1).*(predictedts~=Tslabel) )/(sum(w(n+1:end,1))));
        if( et > 0.5 )
            et = 0.5;
        end
        bt = et/(1-et) + 0.1;
        Bt = [Bt;bt];
        b = 1/( 1 + (2*log(n/N) ));
        predictedtr = svmclassify( svmModel, Trdata );
        temp = ( predictedtr~=Trlabel)';
        temp1 = power(b,temp);
        temp =  ( predictedts ~= Tslabel )';
        temp2 = power(b, -temp);
        w(1:n,1) = w(1:n,1).*temp1';
        w(n+1:end,1) = w(n+1:end,1).*temp2';
    end
    f = size(Trdata,1);
    n1 = size(trtrainX,1);
    cTestX = [];
    cTestY = [];
    for i=1:100
        j = ceil(rand()*n1);
        if( j==0 )
            j = 1;
        end
        cTestX = [ cTestX; trtrainX(j,:)];
        cTestY = [ cTestY; trtrainY(j,:)];
    end
    Y = trAdaBoost(cTestX,svmModels,Bt);
    error = (sum(Y~=cTestY))/size(Y,1);
    Error = [ Error; error];
    fprintf('Accurary on random fold no. %d = %f \n ',fold,1-error );
end
fprintf( ' Average accuracy on 10 random folds = %f \n',1- (sum(Error)/size(Error,1)) ); 
