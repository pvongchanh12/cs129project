%Read compiled dataset, and make column vectors (features)
data = readmatrix("cleanData.csv");

Countries = data(:, 1);
edu = data(:, 2);
inEq = data(:, 3);
womensRights = data(:, 4);
HDI= data(:, 5);

%Feature scaling & mean normalization
eduMean = mean(edu);
eduRange = (max(edu)-min(edu));

inEqMean = mean(inEq);
inEqRange = (max(inEq)-min(inEq));

rightsMean = mean(womensRights);
rightsRange = (max(womensRights)-min(womensRights));

eduAdj = (edu-eduMean)/eduRange;
inEqAdj = (inEq-inEqMean)/inEqRange;
rightsAdj = (womensRights-rightsMean)/rightsRange;

%Making X from features
X = [eduAdj eduAdj.^2 inEqAdj inEqAdj.^2 rightsAdj rightsAdj.^2];

m = size(X, 1);

X = [ones(m, 1) X];

n = size(X, 2);

y = HDI;

testingSet = [];
testingY = [];

%Pick random 30% of data for training set
testSize = round(m*0.3)

%Making X from features
while(size(testingSet, 1) < testSize)
    colNum = randi([1, size(X, 1)])
    testingSet = [testingSet; X(colNum, :)];
    testingY = [testingY; y(colNum, :)];

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

%Gradient descent
for j = 1: numIters
    hypothesis = X*thetas;
    gradient = X'*(hypothesis - y);
    thetas = thetas - (alpha/m)*gradient;
    J_over_time(j) = (hypothesis - y)'*(hypothesis - y)/(2*m);
    iters(j) = j;
end

thetas

%Calculate training and testing cost
trainingCost = (X*thetas - y)'*(X*thetas - y)/(2*m)

testingCost = (testingSet*thetas - testingY)'*(testingSet*thetas - testingY)/(2*size(testingY, 1))

%Plot training J over time: if the model is working, J should decrease with
%each iteration

plot(iters, J_over_time)


