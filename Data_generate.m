clc;
clear;
close all;

Data=xlsread('PV.xlsx');
num1=size(Data,1);
num2=randperm(num1)';
New_PV_Data=Data(num2,:);


% 
% New_PV_Data=New_PV_Data(:,1:end-1);
% D=1+1*randn(3500,8);
% 
% out=D+New_PV_Data;
