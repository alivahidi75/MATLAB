clc;
clear;
close all;

%% Load Data
Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

InputsUN=Data(:,2:end)';
Targets=Data(:,1)';
Targets=full(ind2vec(Targets));

targets=Targets;
inputs=InputsUN;
%% Dovided Data
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

%% Create a and train network

Goal=0;
MaxNeuron=50;
DisplayAT=2;
Spread=2.1;
net=newrb(inputs,targets,Goal,Spread,MaxNeuron,DisplayAT);

%% Test the Network

outputs = net(inputs);
errors = gsubtract(targets,outputs);
performance = perform(net,targets,outputs);

%% Recalculate Training, Validation and Test Performance
%train
trainInputs = inputs(:,trainInd);
trainTargets = targets(:,trainInd);
trainOutputs = outputs(:,trainInd);
trainErrors = trainTargets-trainOutputs;
trainPerformance = perform(net,trainTargets,trainOutputs);

% validation
valInputs = inputs(:,valInd);
valTargets = targets(:,valInd);
valOutputs = outputs(:,valInd);
valErrors = valTargets-valOutputs;
valPerformance = perform(net,valTargets,valOutputs);
% test
testInputs = inputs(:,testInd);
testTargets = targets(:,testInd);
testOutputs = outputs(:,testInd);
testError = testTargets-testOutputs;
testPerformance = perform(net,testTargets,testOutputs);
%%  Plots Results
% View the Network
% view(net);

figure; ploterrhist(errors);
%% confusion matrix
% figure;
% plotconfusion(trainTargets,trainOutputs,'train dadta',...
%               valTargets,valOutputs,'validiation dadta',...
%               testTargets,testOutputs,'tets dadta',...
%               targets,outputs,'all dadta');
          

Targets=vec2ind(targets);
Outputs=vec2ind(outputs);

cm=confusionmat(Targets,Outputs);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;
figure;
cm1 = confusionchart(Targets,Outputs);
MSE=mse(Targets,Outputs);
