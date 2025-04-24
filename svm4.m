clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

InputsUN=Data(:,2:end);
Target=Data(:,1);
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

%% Design SVM
T=zeros(1600,10);
Outputs=zeros(1600,10);
for i=1:10
    t=Target;
    xi=find(t==i);
    t(:)=0;
    t(xi)=1;
    T(:,i)=t;
    Targets=t;
    svmstruct=fitcsvm(Inputs,Targets,'KernelFunction','rbf','Standardize',true,'ClassNames',{'0','1'});
    CVSVMModel = crossval(svmstruct);
    [output,score] = predict(svmstruct,Inputs);
    Outputs(:,i)=str2num(cell2mat(output));
end
CCRtr=zeros(10,1);
MSE=zeros(10,1);
for i=1:10
cm=confusionmat(T(:,i),Outputs(:,i));
CCRtr(i,1)=sum(diag(cm))/(sum(sum(cm)))*100;
figure;
cm1 = confusionchart(T(:,i),Outputs(:,i));
MSE(i,1)=mse(T(:,i),Outputs(:,i));
end
CCRTR=mean(CCRtr);





