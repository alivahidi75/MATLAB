clc;
clear;
close all;

x=linspace(0,2*pi,50);
sigma=0.05;
y=sin(x)+sigma*randn(size(x));
inputs = x; 
targets = y;

nData=size(inputs,2);
perm=randperm(nData);

pTrainData=0.7;
nTrainData=round(pTrainData*nData);
trainInd=perm(1:nTrainData);
perm(1:nTrainData)=[];
trainInputs = inputs(:,trainInd);
trainTargets = targets(:,trainInd);

pTestData=1-pTrainData;
nTestData=nData-nTrainData;
testInd=perm;
testInputs = inputs(:,testInd);
testTargets = targets(:,testInd);

%% 
% Create a and train network
Goal=0;
MaxNeuron=5;
DisplayAT=1;
Spread=1.2;
net=newrb(trainInputs,trainTargets,Goal,Spread,MaxNeuron,DisplayAT);
%%
% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
%%
% Recalculate Training, Validation and Test Performance

trainOutputs = outputs(:,trainInd);
trainErrors = trainTargets-trainOutputs;
trainPerformance = perform(net,trainTargets,trainOutputs);

testOutputs = outputs(:,testInd);
testError = testTargets-testOutputs;
testPerformance = perform(net,testTargets,testOutputs);

PlotResults(targets,outputs,'all dadta');
PlotResults(trainTargets,trainOutputs,'train dadta');
PlotResults(testTargets,testOutputs,'tets dadta');

% View the Network
% view(net);
