clc;
clear;
close all;

%% Load Data

Data=load('Pv_Data');
Data=Data.Pv_Data;

InputsUN=Data(:,2:end);
Targets=Data(:,1)';
Targets=full(ind2vec(Targets));

%% creat zero mean for inputs 

meanIn=mean(InputsUN);

p=size(InputsUN,1);
q=size(InputsUN,2);

b=zeros(p,q);

for i=1:p
   for j=1:q
       b(i,j)=InputsUN(i,j)-meanIn(j);
   end
end
 c1=round(mean(b));
%% PCA
X=b';

[V,Y,S]=pca(X);

inputs=V';
targets=Targets;
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

%% RBF NetWork

Goal=0;
MaxNeuron=9;
DisplayAT=1;
Spread=2.3;
net_rbf=newrb(inputs,targets,Goal,Spread,MaxNeuron,DisplayAT);

%% Test the Network

outputs_rbf = net_rbf(inputs);
errors_rbf = gsubtract(targets,outputs_rbf);
performance_rbf = perform(net_rbf,targets,outputs_rbf);

%% MLP NetWorks
hiddenLayerSize = 5;
TF={'tansig','purelin'};
net_mlp= newff(inputs,targets,hiddenLayerSize,TF);

net_mlp.inputs{1}.processFcns = {'removeconstantrows','mapminmax'};
net_mlp.outputs{2}.processFcns = {'removeconstantrows','mapminmax'};

net_mlp.divideFcn = 'dividerand';  % Divide data randomly
net_mlp.divideMode = 'sample';  % Divide up every sample

net_mlp.trainFcn = 'trainlm';  % Levenberg-Marquardt

net_mlp.performFcn = 'mse';  % Mean squared error

% Choose Plot Functions

net_mlp.plotFcns = {'plotperform','ploterrhist','plotregression','plotfit'};
net_mlp.trainParam.showWindow=false;
net_mlp.trainParam.showCommandLine=false;
net_mlp.trainParam.show=1;
net_mlp.trainParam.epochs=500;
net_mlp.trainParam.goal=1e-8;
net_mlp.trainParam.max_fail=20;
% set same weights 
RandStream.setGlobalStream (RandStream ('mrg32k3a','Seed', 1234));

% Train the Network
[net_mlp,tr_mlp] = train(net_mlp,inputs,targets);

% Test the Network
outputs_mlp = net_mlp(inputs);
errors_mlp = gsubtract(targets,outputs_mlp);
performance_mlp = perform(net_mlp,targets,outputs_mlp);

%% LVQ NetWork
%creat LVQ network
net_lvq=lvqnet(18);

net_lvq.divideFcn='dividerand';

% set same weights 
RandStream.setGlobalStream (RandStream ('mrg32k3a','Seed', 1234));

%Train Network using LVQ1 
net_lvq.trainParam.epochs=20;
net_lvq.trainParam.max_fail=6;
net_lvq=train(net_lvq,inputs,targets);

%Train Network using LVQ2.1 
net_lvq.inputWeights{1}.learnFcn='learnlv2';
net_lvq.trainParam.epochs=30;
net_lvq.trainParam.max_fail=4;
[net_lvq,tr_lvq]=train(net_lvq,inputs,targets);

%Apply Network
outputs_lvq=net_lvq(inputs);
errors_lvq = gsubtract(targets,outputs_lvq);
%% Ensemble Learning

outputs1=vec2ind(round(outputs_rbf))';
outputs2=vec2ind(round(outputs_mlp))';
outputs3=vec2ind(round(outputs_lvq))';

e1=outputs1-outputs2;
ind12=find(e1~=0);
e2=outputs1-outputs3;
ind23=find(e2~=0);
e3=outputs2-outputs3;
ind13=find(e3~=0);
ind=[ind12;ind23;ind13];
ind=sort(ind);
ind=unique(ind);

output=[outputs3,outputs2,outputs1];
outputs=zeros;
 for ii=1:size(output)
    outputs(ii,1)=mode(output(ii,:));
end
outputs=full(ind2vec(outputs'));

%Train Data
TrainInputs=inputs(:,trainInd);
TrainTargets=targets(:,trainInd);
TrainOutputs=outputs(:,trainInd);

%Validation Data
valInputs=inputs(:,valInd);
valTargets=targets(:,valInd);
valOutputs=outputs(:,valInd);
%Test Data
TestInputs=inputs(:,testInd);
TestTargets=targets(:,testInd);
TestOutputs=outputs(:,testInd);

errors = gsubtract(targets,outputs);
%% plot Results

figure;
plotconfusion(TrainTargets,TrainOutputs,'Train',...
    valTargets,valOutputs,'Validation',...
    TestTargets,TestOutputs,'Test',...
    targets,outputs,'All');