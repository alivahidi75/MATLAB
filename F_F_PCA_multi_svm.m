clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

InputsUN=Data(:,2:end);
Targets=Data(:,1);
% Targets=full(ind2vec(Targets));

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

[V,y,S]=pca(X);

Inputs=V;
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