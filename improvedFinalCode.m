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

polynomialFeatures = 6;

%lambdaSet = [0.03];
%polynomialFeatures = 3;

polySet = 1:polynomialFeatures;

allTrainErr = zeros(size(lambdaSet, 2), polynomialFeatures);
allCVErr = zeros(size(lambdaSet, 2), polynomialFeatures);
allTestErr = zeros(size(lambdaSet, 2), polynomialFeatures);


%Making X from features

%Pick random 20% of data for Cross-Validation set and random 20% for
%Testing Set
    trainSize = m;
    testSize = round(m*0.2)
    cvSize = round(m*0.2)

    trainRows = [1:m]'
    cvRows = [];
    testRows = [];
        
        %Select rows of data (examples) for CV set and testing set. Remove
        %those rows from training set
         while(size(cvRows, 1) < cvSize)
            colNum = randi([1, size(trainRows, 1)])
            cvRows = [cvRows; trainRows(colNum)];
        
            trainRows = [trainRows(1:(colNum-1)); trainRows((colNum+1):end)];
         end

        while(size(testRows, 1) < testSize)
            colNum = randi([1, size(trainRows, 1)])
            testRows = [testRows; trainRows(colNum)];
        
            trainRows = [trainRows(1:(colNum-1)); trainRows((colNum+1):end)];
        end

%Iterate over all combos of polynomial degree and lambda value
for l = 1:polynomialFeatures
    for i = 1:size(lambdaSet, 2)
        degree = l;
    
        X = [];

        %Create dataset with newly created polynomial features
        for k = 1:degree
            X = [X feat1.^(k) feat2.^(k) feat3.^(k) feat4.^(k) feat5.^(k) feat6.^(k) feat7.^(k) feat8.^(k) feat9.^(k) feat10.^(k)];
        end
        
        

        m = size(X, 1);
        
        X = [ones(m, 1) X];
        
        n = size(X, 2);
        
        
        %Create individual sets using row numbers
        testingSet = X(testRows, :);
        testingY = HDI(testRows, :);
        
        cvSet = X(cvRows, :);
        cvY = HDI(cvRows, :);
        
        X = X(trainRows, :);
        y = HDI(trainRows, :);
        
        
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
        
        I = eye(size(X'*X));
        I(1, :) = 0;
        thetas = inv([X'*X+lambda*I])*X'*y;

        
        %Calculate training and testing cost
        trainingCost = (X*thetas - y)'*(X*thetas - y)/(2*m);
        
        trainingErrOverL(i) = trainingCost;
    
        testingCost = (testingSet*thetas - testingY)'*(testingSet*thetas - testingY)/(2*size(testingY, 1));
        
        cvCost = (cvSet*thetas - cvY)'*(cvSet*thetas - cvY)/(2*size(cvY, 1));
    
        cvErrOverL(i) = cvCost;
    
        testingErrOverL(i) = testingCost;
    
        %Plot training J over time: if the model is working, J should decrease with
        %each iteration
        
        allTrainErr(i, l) = trainingCost;
        allCVErr(i, l) = cvCost;
        allTestErr(i, l) = testingCost;
    end
    
    
    %Now that we know lambda, final model:
    
        lambda = 0.03;
        
        while(size(testingSet, 1) < testSize)
            colNum = randi([1, size(X, 1)]);
            testingSet = [testingSet; X(colNum, :)];
            testingY = [testingY; y(colNum, :)];
        
            X = [X(1:(colNum-1), :); X((colNum+1):end, :)];
            y = [y(1:(colNum-1), :); y((colNum+1):end, :)];
        
        end
        
        %Making X from features
        while(size(cvSet, 1) < cvSize)
            colNum = randi([1, size(X, 1)]);
            cvSet = [cvSet; X(colNum, :)];
            cvY = [cvY; y(colNum, :)];
        
            X = [X(1:(colNum-1), :); X((colNum+1):end, :)];
            y = [y(1:(colNum-1), :); y((colNum+1):end, :)];
        
        end
        
        I = eye(size(X'*X))
        I(1, :) = 0;
        thetas = inv([X'*X+lambda*I])*X'*y
    
        
        %Calculate training, testing, and CV cost
        trainingCost = (X*thetas - y)'*(X*thetas - y)/(2*m);
    
        testingCost = (testingSet*thetas - testingY)'*(testingSet*thetas - testingY)/(2*size(testingY, 1));
        
        cvCost = (cvSet*thetas - cvY)'*(cvSet*thetas - cvY)/(2*size(cvY, 1));
    
    
    
    
    ErrorsAll = [trainingErrOverL; cvErrOverL; testingErrOverL];
    end
end

%plot(lambdaSet, trainingErrOverL)
%hold on
%plot(lambdaSet, cvErrOverL)
%plot(lambdaSet, testingErrOverL)
%xlabel("Lambda value")
%ylabel("Error")
%legend("Training Set","Cross-Validation Set", "Testing Set")

surf(lambdaSet, polySet, allCVErr')

title("Cross-validation error versus lambda and polynomal degree")

xlabel("Lambda value")
ylabel("Degree of polynomial")
zlabel("Cross-validation Error")

a = [allTrainErr(1, :)'; allTrainErr(2, :)'; allTrainErr(3, :)'; allTrainErr(4, :)';allTrainErr(5, :)';allTrainErr(2, :)';allTrainErr(6, :)'];
b = [allTestErr(1, :)'; allTestErr(2, :)'; allTestErr(3, :)'; allTestErr(4, :)';allTestErr(5, :)';allTestErr(2, :)';allTestErr(6, :)'];

c = [a b]

writematrix(c, "thisMat.csv");

groundTruthComparison = [testingSet*thetas testingY];
