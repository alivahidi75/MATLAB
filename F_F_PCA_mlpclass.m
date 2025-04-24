clc;
clear;
close all;

%% Load Data
Data=xlsread('Book2.xlsx');
Data2 = xlsread('Book3.xlsx');
X = Data(:,1:7);
Y = Data(:, 8);

%% Create and train Network
hiddenLayerSize1 = 20;
hiddenLayerSize2 = 15;
TF={'tansig','tansig'};
net = newff(X,Y,[hiddenLayerSize1 hiddenLayerSize2],TF);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess

net.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net.targets{2}.processFcns = {'removeconstantrows','mapminmax'};


% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide

net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
RandStream.setGlobalStream (RandStream ('mrg32k3a','Seed', 1234));
% For help on training function 'trainlm' type: help trainlm
% For a list of all training functions type: help nntrain

net.trainFcn = 'trainlm';  % Levenberg-Marquardt

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','ploterrhist','plotregression','plotfit'};

net.trainParam.showWindow=false;
net.trainParam.showCommandLine=false;
net.trainParam.show=1;
net.trainParam.epochs=500;
net.trainParam.goal=1e-8;
net.trainParam.max_fail=20;

% Train the Network
[net,tr] = train(net,X,Y);

% Test the Network
outputs = net(X);
errors = gsubtract(Y,outputs);
performance = perform(net,Y,outputs);
%% Recalculate Training, Validation and Test Performance
trainInd=tr.trainInd;
trainInputs = X(:,trainInd);
trainTargets = Y(:,trainInd);
trainOutputs = outputs(:,trainInd);
trainErrors = trainTargets-trainOutputs;
trainPerformance = perform(net,trainTargets,trainOutputs);

valInd=tr.valInd;
valInputs = X(:,valInd);
valTargets = Y(:,valInd);
valOutputs = outputs(:,valInd);
valErrors = valTargets-valOutputs;
valPerformance = perform(net,valTargets,valOutputs);

testInd=tr.testInd;
testInputs = X(:,testInd);
testTargets = Y(:,testInd);
testOutputs = outputs(:,testInd);
testError = testTargets-testOutputs;
testPerformance = perform(net,testTargets,testOutputs);
%% Results Plots
PlotResults(Y,outputs,'all dadta');
PlotResults(trainTargets,trainOutputs,'train dadta');
PlotResults(valTargets,valOutputs,'validiation dadta');
PlotResults(testTargets,testOutputs,'tets dadta');

