clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

InputsUN=Data(:,2:end);
Targets=Data(:,1)';
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

%% knn classifire
X=Inputs;
Y=Targets';
% c=ClassificationKNN.fit(x,y,'NumNeighbors',5);
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
Z_knn = resubPredict(Mdl);

resubLoss(Mdl);
disp('Reub-Loss_knn:');
disp(resubLoss(Mdl));

%% cross val
cvmodel=crossval(Mdl);
kfoldLoss(cvmodel);
disp('kfoldLoss_knn:');
disp(kfoldLoss(cvmodel));
%% Display
cm_knn=confusionmat(Y,Z_knn);
CCRtr_knn=sum(diag(cm_knn))/(sum(sum(cm_knn)))*100;
figure;
cm1_knn = confusionchart(Y,Z_knn);
MSE_KNN=mse(Y,Z_knn);