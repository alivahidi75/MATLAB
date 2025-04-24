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

%%  Naive Bayesian classifire
X=Inputs;
Y=Targets;

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



