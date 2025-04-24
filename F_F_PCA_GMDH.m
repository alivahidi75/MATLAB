clc;
clear;
close all;

%% Load Data

Data=load('New_Pv_Data');
Data=Data.New_PV_Data;

Inputs=Data(:,2:end)';

Target=Data(:,1)';
% Targets=full(ind2vec(Targets));
T=zeros(10,1600);
O=zeros(10,1600);
for i=1:10
t=Target;
xi=find(t==i);
t(:)=1;
t(xi)=0;
T(i,:)=t;
Targets=t;

nData = size(Inputs,2);
Perm = randperm(nData);

% Train Data
pTrain = 0.9;
nTrainData = round(pTrain*nData);
TrainInd = Perm(1:nTrainData);
TrainInputs = Inputs(:,TrainInd);
TrainTargets = Targets(:,TrainInd);

% Test Data
pTest = 1 - pTrain;
nTestData = nData - nTrainData;
TestInd = Perm(nTrainData+1:end);
TestInputs = Inputs(:,TestInd);
TestTargets = Targets(:,TestInd);


params.MaxLayerNeurons = 15;   % Maximum Number of Neurons in a Layer
params.MaxLayers = 5;          % Maximum Number of Layers
params.alpha = 0.6;            % Selection Pressure
params.pTrain = 0.7;           % Train Ratio
gmdh = GMDH(params, TrainInputs, TrainTargets);


Outputsi = ApplyGMDH(gmdh, Inputs);
Outputsi = double(Outputsi>=0.5);
O(i,:)=Outputsi;
TrainOutputs = Outputsi(:,TrainInd);
TestOutputs = Outputsi(:,TestInd);
clear t
clear Targets
end

%% Show Results

% for i=1:10
% figure(i);
% plotconfusion(T(i,:),O(i,:), 'All Data');
% end
%% confusion matrix
figure(1);
subplot(2,5,1)
plotconfusion(T(1,:),O(1,:), 'All Data Normal');
figure(2);
subplot(2,5,2)
plotconfusion(T(2,:),O(2,:), 'All Data LL 2 to 4');
figure(3);
subplot(2,5,3)
plotconfusion(T(3,:),O(3,:), 'All Data LL 50%');
figure(4);
subplot(2,5,4)
plotconfusion(T(4,:),O(4,:), 'All Data OC1');
figure(5);
subplot(2,5,5)
plotconfusion(T(5,:),O(5,:), 'All Data PS 65%');
figure(6);
subplot(2,5,6)
plotconfusion(T(6,:),O(6,:), 'All Data DP');
figure(7);
subplot(2,5,7)
plotconfusion(T(7,:),O(7,:), 'All Data SCD2');
figure(8);
subplot(2,5,8)
plotconfusion(T(8,:),O(8,:), 'All Data LL3');
figure(9);
subplot(2,5,9)
plotconfusion(T(9,:),O(9,:), 'All Data PS 80%');
figure(10);
subplot(2,5,10)
plotconfusion(T(10,:),O(10,:), 'All Data SCD1');