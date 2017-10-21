%% FLAP
% Chen Gong, Dacheng Tao, Keren Fu, Jie Yang. Fick¡¯s Law Assisted
% Propagation for semi-supervised learning. IEEE Transactions on Neural
% Networks and Learning Systems (TNNLS), 2015, 26(9): 2148-2162.

%%
clear all
close all;
clc;


%% Parameters
sigma                  = 0.5;          % Kernel width
K                      = 10;           % Nearest neighbors
Alpha                  = 0.99;         % trade-off parameter
yita                   = 0.1;          % eta

%% Load Dataset
load Iris.mat;
Data = Iris.Data;
GroundTruth = Iris.GroundTruth;
LabeledIndex = Iris.LabeledIndex;
UnlabeledIndex = Iris.UnlabeledIndex;
[DataTotal,Dim] = size(Data);
ClassTotal  = max(GroundTruth);            % number of classes

%% Construct graph
W = ConstructKNNGraph(Data, sigma, K);
[P, gamma] = ComputeP(W, yita);

%% Label Propagation
Y = zeros(DataTotal,ClassTotal);
for i=1:DataTotal
    Y(i,GroundTruth(i))=1;
end

F = FLAPLabelPropagation(P, Y, Alpha);


%% Output class information
[~,Classification] = max(F,[],2);

%% evaluation
classes = (1:ClassTotal)';
[confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth,Classification,classes);

display(['accuracy = ',num2str(Accuracy)])













