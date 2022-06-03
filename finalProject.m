%Read compiled dataset, and make column vectors (features)

clf;
theseCountries = readmatrix("theseCountries.csv");
data = readmatrix("improvedSet.csv");

HDI = data(:, 2);

%Feature scaling & mean normalization

for i = 3:(size(data, 2))
    feat = data(:, i)
    featMean = mean(feat)
    featRange = (max(feat)-min(feat));

    data(:, i) = (data(:, i)-featMean)/featRange
end

feat1 = data(:, 3);
feat2 = data(:, 4);
feat3 = data(:, 5);
feat4 = data(:, 6);
feat5 = data(:, 7);
feat6 = data(:, 8);
feat7 = data(:, 9);
feat8 = data(:, 10);
feat9 = data(:, 11);
feat10 = data(:, 12);

lambdaSet = [0.001 0.003 0.01 0.03 0.1 0.3 1];

polynomialFeatures = 5

%Making X from features
X = [];

for i = 1:polynomialFeatures
    X = [X feat1.^(i) feat2.^(i) feat3.^(i) feat4.^(i) feat5.^(i) feat6.^(i) feat7.^(i) feat8.^(i) feat9.^(i) feat10.^(i)]
end

m = size(X, 1);

X = [ones(m, 1) X];

n = size(X, 2);

y = HDI;

testingSet = [];
testingY = [];

cvSet = [];
cvY = [];

%Pick random 30% of data for training set
testSize = round(m*0.2)
cvSize = round(m*0.2)


trainingErrOverL = zeros(size(lambdaSet));
cvErrOverL = zeros(size(lambdaSet));
testingErrOverL = zeros(size(lambdaSet));


for i = 1:size(lambdaSet, 2)

    lambda = lambdaSet(i);
    
    while(size(testingSet, 1) < testSize)
        colNum = randi([1, size(X, 1)])
        testingSet = [testingSet; X(colNum, :)];
        testingY = [testingY; y(colNum, :)];
    
        X = [X(1:(colNum-1), :); X((colNum+1):end, :)];
        y = [y(1:(colNum-1), :); y((colNum+1):end, :)];
    
    end
    
    %Making X from features
    while(size(cvSet, 1) < cvSize)
        colNum = randi([1, size(X, 1)])
        cvSet = [cvSet; X(colNum, :)];
        cvY = [cvY; y(colNum, :)];
    
        X = [X(1:(colNum-1), :); X((colNum+1):end, :)];
        y = [y(1:(colNum-1), :); y((colNum+1):end, :)];
    
    end
    
    
    %Picking step size and number of iterations
    alpha = 0.03
    
    numIters = 1000;
    
    %Make vector of training J over iterations
    

    J_over_time = zeros(numIters, 1);
    iters = zeros(numIters, 1);
    
    thetas = zeros(n,1);
    
    thetas = inv([X'*X+lambda*eye(size(X'*X))])*X'*y

    %Gradient descent
    %for j = 1: numIters
        
        %hypothesis = X*thetas;
        %gradient = X'*(hypothesis - y);
        %thetas = thetas - (alpha/m)*gradient;
        %J_over_time(j) = (hypothesis - y)'*(hypothesis - y)/(2*m);
        %iters(j) = j;
    %end
    
    
    %Calculate training and testing cost
    trainingCost = (X*thetas - y)'*(X*thetas - y)/(2*m)
    
    trainingErrOverL(i) = trainingCost;

    testingCost = (testingSet*thetas - testingY)'*(testingSet*thetas - testingY)/(2*size(testingY, 1))
    
    cvCost = (cvSet*thetas - cvY)'*(cvSet*thetas - cvY)/(2*size(cvY, 1))

    cvErrOverL(i) = cvCost;

    testingErrOverL(i) = testingCost;

    %Plot training J over time: if the model is working, J should decrease with
    %each iteration
    
    plot(iters, J_over_time)
end


%Now that we know lambda, final model:

    lambda = 0.03;
    
    while(size(testingSet, 1) < testSize)
        colNum = randi([1, size(X, 1)])
        testingSet = [testingSet; X(colNum, :)];
        testingY = [testingY; y(colNum, :)];
    
        X = [X(1:(colNum-1), :); X((colNum+1):end, :)];
        y = [y(1:(colNum-1), :); y((colNum+1):end, :)];
    
    end
    
    %Making X from features
    while(size(cvSet, 1) < cvSize)
        colNum = randi([1, size(X, 1)])
        cvSet = [cvSet; X(colNum, :)];
        cvY = [cvY; y(colNum, :)];
    
        X = [X(1:(colNum-1), :); X((colNum+1):end, :)];
        y = [y(1:(colNum-1), :); y((colNum+1):end, :)];
    
    end
    
    
    thetas = inv([X'*X+lambda*eye(size(X'*X))])*X'*y

    
    %Calculate training and testing cost
    trainingCost = (X*thetas - y)'*(X*thetas - y)/(2*m)

    testingCost = (testingSet*thetas - testingY)'*(testingSet*thetas - testingY)/(2*size(testingY, 1))
    
    cvCost = (cvSet*thetas - cvY)'*(cvSet*thetas - cvY)/(2*size(cvY, 1))





ErrorsAll = [trainingErrOverL; cvErrOverL; testingErrOverL];

plot(lambdaSet, trainingErrOverL)
hold on
plot(lambdaSet, cvErrOverL)
plot(lambdaSet, testingErrOverL)


