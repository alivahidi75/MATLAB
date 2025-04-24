clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

Inputs=Data(:,2:end);
Targets=Data(:,1);
% Targets=full(ind2vec(Targets));

%% multiclassification with SVM

t = templateSVM('Standardize',true,'SaveSupportVectors',true);
Mdl = fitcecoc(Inputs,Targets,'Learners',t);
CVMdl = crossval(Mdl);

%% predict
Outputs=predict(Mdl,Inputs);
%% Results
cm=confusionmat(Targets,Outputs);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;
figure;
cm1 = confusionchart(Targets,Outputs);
MSE=mse(Targets,Outputs);

