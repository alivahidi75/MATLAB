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
Y=Targets;
% c=ClassificationKNN.fit(x,y,'NumNeighbors',5);
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
Z_knn = resubPredict(Mdl);

resubLoss(Mdl);
disp('Reub-Loss_knn:');
disp(resubLoss(Mdl));
%%
cm=confusionmat(Y,Z_knn);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;
figure;
cm1 = confusionchart(Y,Z_knn);
MSE=mse(Y,Z_knn);