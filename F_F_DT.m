clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

Inputs=Data(:,2:end);
Targets=Data(:,1)';
% Targets=full(ind2vec(Targets));

%% Decision Tree
X=Inputs;
Y=Targets;
t=ClassificationTree.fit(X,Y);
% view(t,'mode','graph');
e=resubLoss(t);
disp('resubLoss_dt:');
disp(e)
%% cross val
 cv=crossval(t);
 e1=kfoldLoss(cv);
 disp('kfoldLoss_dt:');
 disp(e1);
 
 %% test
 Z_dt=predict(t,X);
 %% Display
cm_dt=confusionmat(Y,Z_dt);
CCRtr_dt=sum(diag(cm_dt))/(sum(sum(cm_dt)))*100;
figure;
cm1_dt = confusionchart(Y,Z_dt);
MSE_DT=mse(Y,Z_dt);