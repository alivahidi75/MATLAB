clc;
clear;
close all;
%% Load Data
x=linspace(0,2*pi,200);
sigma=0;
y=sin(x)+sigma*randn(size(x));

inputs = x; 
targets = y;

nData=size(inputs,2);
perm=randperm(nData);

pTrainData=0.7;
nTrainData=round(pTrainData*nData);
trainInd=perm(1:nTrainData);
perm(1:nTrainData)=[];

pTestData=0.15;
nTestData=round(pTestData*nData);
testInd=perm(1:nTestData);
perm(1:nTestData)=[];

pValData=1-pTrainData-pTestData;
nValData=nData-nTrainData-nTestData;
valInd=perm;
%% 
% Create a and train network
Goal=0;
MaxNeuron=5;
DisplayAT=1;
Spread=1.2;
net=newrb(inputs,targets,Goal,Spread,MaxNeuron,DisplayAT);
%%
% Test the Network
outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);
%%
% Recalculate Training, Validation and Test Performance
%train
% trainInd=tr.trainInd;
% trainInputs = inputs(:,trainInd);
% trainTargets = targets(:,trainInd);
% trainOutputs = outputs(:,trainInd);
% trainErrors = trainTargets-trainOutputs;
% trainPerformance = perform(net,trainTargets,trainOutputs);

% valInd=tr.valInd;
% valInputs = inputs(:,valInd);
% valTargets = targets(:,valInd);
% valOutputs = outputs(:,valInd);
% valErrors = valTargets-valOutputs;
% valPerformance = perform(net,valTargets,valOutputs);
% 
% testInd=tr.testInd;
% testInputs = inputs(:,testInd);
% testTargets = targets(:,testInd);
% testOutputs = outputs(:,testInd);
% testError = testTargets-testOutputs;
% testPerformance = perform(net,testTargets,testOutputs);

PlotResults(targets,outputs,'all dadta');
% PlotResults(trainTargets,trainOutputs,'train dadta');
% PlotResults(valTargets,valOutputs,'validiation dadta');
% PlotResults(testTargets,testOutputs,'tets dadta');

% View the Network
% view(net);

% Plots
% Uncomment these lines to enable various plots.

% figure;
% plotperform(tr);

% figure;
% plottrainstate(tr);

% figure;
% plotfit(net,inputs,targets);

% figure;
% plotregression(trainTargets,trainOutputs,'Train Data',...
%     valTargets,valOutputs,'Validation Data',...
%     testTargets,testOutputs,'Test Data',...
%     targets,outputs,'All Data')

% figure;
% ploterrhist(errors);

