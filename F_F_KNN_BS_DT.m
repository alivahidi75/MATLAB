clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

InputsUN=Data(:,2:end);
Targets=Data(:,1)';
% Targets=full(ind2vec(Targets));

%% knn classifire
X=InputsUN;
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
%%  Naive Bayesian classifire

bc=fitcnb(X,Y);
Z_nb=bc.predict(X);
e=resubLoss(bc);
disp('resubLoss_nb:');
disp(e)
%% cross val
 cv=crossval(bc);
 e1=kfoldLoss(cv);
 disp('kfoldLoss_nb:');
 disp(e1);
%% Display

cm_nb=confusionmat(Y,Z_nb);
CCRtr_nb=sum(diag(cm_nb))/(sum(sum(cm_nb)))*100;
figure;
cm1_nb = confusionchart(Y,Z_nb);
MSE_NB=mse(Y,Z_nb);
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
MSE_DT=mse(Y,Z_dt);
%% Ensembel learning
Z=[Z_knn Z_dt Z_nb];
z_s=zeros;
for ii=1:size(Z)
    z_s(ii,1)=mode(Z(ii,:));
end
cm=confusionmat(Y,z_s);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;
figure;
cm1 = confusionchart(Y,z_s);
MSE=mse(Y,z_s);
 %% Display

