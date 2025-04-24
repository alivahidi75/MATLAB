clc;
clear;
close all;

%% Load Data

Data=load('Pv_Data');
Data=Data.Pv_Data;

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
%% Decision Tree
X=Inputs;
Y=Targets';

t=ClassificationTree.fit(X,Y);
% view(t,'mode','graph');
e=resubLoss(t);
disp('resubLoss=');
disp(e)
%% cross val
 cv=crossval(t);
 e1=kfoldLoss(cv);
 disp('kfoldLoss=');
 disp(e1);
 
 %% test
 Z=predict(t,X);
 %% Display
cm=confusionmat(Y,Z);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;
cm1 = confusionchart(Y,Z);