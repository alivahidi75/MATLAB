clc;
clear;
close all;

%% Load Data

Data=xlsread('New_PV_Data3.xlsx');
Data=Data(:,9:17);
num1=size(Data,1);
num2=randperm(num1)';
New_PV_Data3=Data(num2,:);
%%
Data=load('New_Pv_Data3');
Data=Data.New_PV_Data3;

Inputs=Data(:,1:end-1)';
Targets=Data(:,end);

meanIn=mean(Inputs);

p=size(Inputs,1);
q=size(Inputs,2);

b=zeros(p,q);

for i=1:p
   for j=1:q
       b(i,j)=Inputs(i,j)-meanIn(j);
   end
end
 c1=round(mean(b));
%% PCA
X=b;

[V,Y,S]=pca(X);

Inputs=V;

%% 
nData=size(Inputs,1);
pTrain=0.7;
nTrain=round(pTrain*nData);
TrainInputs=Inputs(1:nTrain,:);
TrainTargets=Targets(1:nTrain);
TrainData = [TrainInputs TrainTargets];

pTest=1-pTrain;
nTest=nData-nTrain;
TestInputs=Inputs(nTrain+1:end,:);
TestTargets=Targets(nTrain+1:end);

%% Design ANFIS
%genfis1
% nMFs=3;
% InputMF='trimf';
% OutputMF='constant';

%fis = genfis1(TrainData,nMFs,InputMF,OutputMF);
%fis = genfis2(TrainInputs,TrainTargets,0.7); 
fis=genfis3(TrainInputs,TrainTargets,'sugeno',16);

MaxEpoch=150;
ErrorGoal=0.01;
InitialStepSize=0.01;
StepSizeDecreaseRate=0.1;
StepSizeIncreaseRate=1.2;
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid
            
fis=anfis(TrainData,fis,TrainOptions,DisplayOptions,[],OptimizationMethod);


%% Apply ANFIS to Train Data

TrainOutputs=evalfis(TrainInputs,fis);
TrainOutputs = round(TrainOutputs);

% for i=1:numel(TrainOutputs)
%     if TrainOutputs(i)==0
%         TrainOutputs(i)=2;
%     end
% end
TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors(:).^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);
MSE=mse(TrainTargets,TrainOutputs);

cm=confusionmat(TrainTargets,TrainOutputs);
CCRtr=sum(diag(cm))/(sum(sum(cm)))*100;

for i=1:21
    TP(i)=cm(i,i);
    FP(i)=sum(cm(i,:))-TP(i);
    FN(i)=sum(cm(:,i))-TP(i);
    TN(i)=sum(sum(cm))-sum(cm(i,:))-sum(cm(:,i));
    P(i)=TP(i)/(TP(i)+FP(i));
    R(i)=TP(i)/(TP(i)+FN(i));
    FF(i)=2*(P(i)*R(i))/(P(i)+R(i));
end
Per=mean(P);
Rec=mean(R);
F=2*(Per*Rec)/(Per+Rec);

figure;
cm = confusionchart(TrainTargets,TrainOutputs);
cm.Title = 'Train Data Classification';

%% Apply ANFIS to Test Data

TestOutputs=evalfis(TestInputs,fis);
TestOutputs = round(TestOutputs);

for i=1:numel(TestOutputs)
   if TestOutputs(i)==22
       TestOutputs(i)=21;
   elseif TestOutputs(i)==365
           TestOutputs(i)=5;
   end    
end
TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);
MSE1=mse(TrainTargets,TrainOutputs);

cm1=confusionmat(TestTargets,TestOutputs);
CCRtr1=sum(diag(cm1))/(sum(sum(cm1)))*100;

for i=1:21
    TP(i)=cm1(i,i);
    FP(i)=sum(cm1(i,:))-TP(i);
    FN(i)=sum(cm1(:,i))-TP(i);
    TN(i)=sum(sum(cm1))-sum(cm1(i,:))-sum(cm1(:,i));
    P1(i)=TP(i)/(TP(i)+FP(i));
    R1(i)=TP(i)/(TP(i)+FN(i));
    FF1(i)=2*(P1(i)*R1(i))/(P1(i)+R1(i));
end
Per1=mean(P1);
Rec1=mean(R1);
F1=2*(Per1*Rec1)/(Per1+Rec1);

figure;
cm1 = confusionchart(TestTargets,TestOutputs);
cm1.Title = 'Test Data Classification';


%% Apply ANFIS to All Data

Outputs=evalfis(Inputs,fis);
Outputs = round(Outputs);

for i=1:numel(Outputs)
   if Outputs(i)==22
       Outputs(i)=21;
   elseif Outputs(i)==365
           Outputs(i)=5;
   end    
end
Errors=Targets-Outputs;
MSE=mean(Errors(:).^2);
RMSE=sqrt(MSE);
ErrorMean=mean(Errors);
ErrorSTD=std(Errors);
MSE2=mse(Targets,Outputs);

cm2=confusionmat(Targets,Outputs);
CCRtr2=sum(diag(cm2))/(sum(sum(cm2)))*100;
TP=[];
FP=[];
FN=[];
TN=[];
FF2=[];

for i=1:21
    TP(i)=cm2(i,i);
    FP(i)=sum(cm2(i,:))-TP(i);
    FN(i)=sum(cm2(:,i))-TP(i);
    TN(i)=sum(sum(cm2))-sum(cm2(i,:))-sum(cm2(:,i));
    P2(i)=TP(i)/(TP(i)+FP(i));
    R2(i)=TP(i)/(TP(i)+FN(i));
    FF2(i)=2*(P2(i)*R2(i))/(P2(i)+R2(i));
end
Per2=mean(P2);
Rec2=mean(R2);
F2=2*(Per2*Rec2)/(Per2+Rec2);

figure;
cm2 = confusionchart(Targets,Outputs);
cm2.Title = 'All Data Classification';
%cm2.RowSummary = 'row-normalized';
%cm2.ColumnSummary = 'column-normalized';

