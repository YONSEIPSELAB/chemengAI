%% import dataset
StdX = xlsread('StdX.xlsx');
Response = xlsread('RawY.xlsx');
x = StdX';
t = Response';

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainlm';  

%% Create a Fitting Network
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize,trainFcn);

%% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

%% Train the Network
[net,tr] = train(net,x,t);

%% Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);

% View the Network
view(net)

%% Parity plots
figure, plotregression(t,y)

%% Surrogate model handle
genFunction(net,'NN_function','MatrixOnly','yes');

fun = @(x) NN_function(x');

%% Metaheuristic optimization
Numvar = 5;
lowerbound = [0 0 0 0 0];
Upperbound = [1 1 1 1 1];

opts = optimoptions(@ga, 'PlotFcn',{@gaplotbestf});
[x,fval,exitFlag,Output] = ga(fun,Numvar,[],[],[],[],lowerbound,Upperbound,[], opts);
