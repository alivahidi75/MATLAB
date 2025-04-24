clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

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

[V, Y, S]=pca(X);

Inputs=V';

%% Creat and Train Network

%creat LVQ network
net=lvqnet(200);

%Divide Data
pTrain=0.7;
pTest=0.15;
pVal=1-pTrain-pTest;

net.divideFcn='dividerand';
net.divideParam.trainRatio=pTrain;
net.divideParam.valRatio=pVal;
net.divideParam.testRatio=pTest;
% set same weights 
RandStream.setGlobalStream (RandStream ('mrg32k3a','Seed', 1234));

%Train Network using LVQ1 
net.trainParam.epochs=20;
net.trainParam.max_fail=7;
net=train(net,Inputs,Targets);

%Train Network using LVQ2.1 
net.inputWeights{1}.learnFcn='learnlv2';
net.trainParam.epochs=30;
net.trainParam.max_fail=4;
[net,tr]=train(net,Inputs,Targets);

%Apply Network
Outputs=net(Inputs);

%% Data Division 

%Train Data
TrainInputs=Inputs(:,tr.trainInd);
TrainTargets=Targets(:,tr.trainInd);
TrainOutputs=Outputs(:,tr.trainInd);

%Validation Data
valInputs=Inputs(:,tr.valInd);
valTargets=Targets(:,tr.valInd);
valOutputs=Outputs(:,tr.valInd);
%Test Data
TestInputs=Inputs(:,tr.testInd);
TestTargets=Targets(:,tr.testInd);
TestOutputs=Outputs(:,tr.testInd);
%% Plot Results
% figure;
% plotconfusion(TrainTargets,TrainOutputs,'Train',...
%               valTargets,valOutputs,'Validation',...
%               TestTargets,TestOutputs,'Test',...
%               Targets,Outputs,'All');
% figure;
% plotroc(TrainTargets,TrainOutputs,'Train',...
%         valTargets,valOutputs,'Validation',...
%         TestTargets,TestOutputs,'Test',...
%         Targets,Outputs,'All');

%% 
Targets=vec2ind(Targets);
Outputs=vec2ind(Outputs);

cm=confusionmat(Targets,Outputs);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;
figure;
cm1 = confusionchart(Targets,Outputs);
MSE=mse(Targets,Outputs);