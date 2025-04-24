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
title('KNN')
MSE_KNN=mse(Y,Z_knn);
%% multiclassification with SVM
Md2= templateSVM('Standardize',true,'SaveSupportVectors',true);
Mdl = fitcecoc(Inputs,Targets,'Learners',Md2);
%% cross val
CVMdl = crossval(Mdl);
 e2=kfoldLoss(CVMdl);
 disp('kfoldLoss_svm:');
 disp(e2);
%% predict
Z_SVM=predict(Mdl,X);
%% Results
cm_svm=confusionmat(Y,Z_SVM);
CCRtr_svm=sum(diag(cm_svm))/(sum(sum(cm_svm)))*100;
figure;
cm1_svm = confusionchart(Y,Z_SVM);
title('SVM')
MSE_svm=mse(Y,Z_SVM);
%% Decision Tree

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
title('DT')
MSE_DT=mse(Y,Z_dt);
%% Ensembel learning
Z=[Z_knn Z_dt Z_SVM];
z_s=zeros;
for ii=1:size(Z)
    z_s(ii,1)=mode(Z(ii,:));
end
cm=confusionmat(Y,z_s);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;
figure;
cm1 = confusionchart(Y,z_s);
title('Ensemble')
MSE=mse(Y,z_s);
