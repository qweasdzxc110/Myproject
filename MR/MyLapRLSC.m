function [results,meanL, stdL]=MyLapRLSC(dataset, NN,DEGREE,SIGMA) 
% [err_svm,err_rlsc]=Experiments(dataset)
% dataset.mat should have matrices 
% X (n x d training data matrix, n examples, d features)
% Xt (test data matrix) (if available)     
% training labels Y      
% test labels Yt (if available)
%
% NN: number of nearest neighbors for graph laplacian computation
% SIGMA: width of RBF kernel
% gamma_A,gamma_I: the extrinsic and intrinsic regularization parameters
% DEGREE: power of the graph Laplacian to use as the graph regularizer
% M: Precomputed grah regularizer if available (n x n matrix).
%
% This code generates the mean results training and testing on data
% splits.
% 
% Data splits are specified as follows -- 
% folds: a n x R matrix with 0 or 1 values; each column indicates which
% examples are to be considered labeled (1) or unlabeled (0). 
% Alternatively, one can specify by a collection of indices in R x l matrix 
% idxLabs where each row gives the l indices for labeled examples. 
%
% For experiments in the paper "Beyond the Point Cloud", run e.g.,
% 
% Experiments('USPST'); % USPST.mat contains all the options
%% 
if strcmp(dataset, 'TDT2.mat')
        load 'TDT2.mat';
        C=unique(gnd);
        X = tfidf(fea);
        DataStruct.GroundTruth = gnd;
       % [row, col] = size(gnd);
%         Data.LabeledIndex = foldslabel(gnd, labelrate);
        
        %Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
        number_runs = 10;
end
if strcmp(dataset, 'breast.mat')
        load 'breast.mat';
        C=unique(label);
        X = NormalizeFea(data);
        DataStruct.GroundTruth = label;
       % [row, col] = size(gnd);
%         Data.LabeledIndex = foldslabel(gnd, labelrate);
        %Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
        number_runs = 10;
end
if strcmp(dataset, 'glass.mat')
        load 'glass.mat';
        C=unique(label);
        X = NormalizeFea(data);
        DataStruct.GroundTruth = label;
        number_runs = 10;
end
if strcmp(dataset, 'USPST.mat')
    load(dataset);
    C=unique(Y);
    number_runs=size(idxLabs,1); 
    DataStruct.GroundTruth = Y;
end

if length(C)==2
             C=[1 -1]; nclassifiers=1;
else
               nclassifiers=length(C);
end
% options contains the optimal parameter settings for the data 
options=make_options;
options.NN=NN;
options.Kernel='rbf';
options.KernelParam=SIGMA;
options.gamma_A=1e-06; 
% options.gamma_A = gamma_A;
options.gamma_I=0.01; 
% options.gamma_I=gamma_I;
options.GraphWeights='heat';
options.GraphWeightParam=SIGMA;
options.LaplacianDegree=DEGREE;
M=laplacian(options,X);
M=M^options.LaplacianDegree;
tic;
% construct and deform the kernel
% K contains the gram matrix of the warped data-dependent semi-supervised kernel
G=calckernel(options,X);
r=options.gamma_I/options.gamma_A;
% the deformed kernel matrix evaluated over test data
% fprintf(1,'Deforming Kernel\n');
K=Deform(r,G,M);

% run over the random splits
% for foldi = 1:number_runs
%     folds(foldi,:)=foldslabel(DataStruct.GroundTruth,rate);
% end 
% if exist('folds','var')
%    number_runs=size(folds,2);
% else
%    number_runs=size(idxLabs,1); 
% end
for foldi = 1:number_runs
    folds(foldi,:) = genlabidx(DataStruct.GroundTruth,2);
end
for R=1:number_runs
% 	if exist('folds','var')
%       L=find(folds(R,:));
%     else
% %       L=idxLabs(R,:);
%     end
    L=folds(R,:);
% L=idxLabs(R,:);
    U=(1:size(K,1))'; 
    U(L)=[];
	data.K=K(L,L); data.X=X(L,:); 
    frlsc=[];
 
    for c=1:nclassifiers
        if nclassifiers>1
            fprintf(1,'Class %d versus rest\n',C(c)); 
        end
        data.Y=(DataStruct.GroundTruth(L)==C(c))-(DataStruct.GroundTruth(L)~=C(c)); % labeled data
%         data.Y = DataStruct.GroundTruth(L);
        classifier_rlsc=rlsc(options,data);
         
        frlsc(:,c)=K(U,L(classifier_rlsc.svs))*classifier_rlsc.alpha-classifier_rlsc.b;
        if exist('bias','var')
          [frlsc(:,c),classifier_rlsc.b] = adjustbias(frlsc(:,c)+classifier_rlsc.b,bias);
        end
 
        results(R).frlsc(:,c)=frlsc(:,c); 
        yu=(DataStruct.GroundTruth(U)==C(c))-(DataStruct.GroundTruth(U)~=C(c));

    end
   if nclassifiers==1
        frlsc=sign(results(R).frlsc);
        if exist('Xt','var')
          frlsc_t=sign(results(R).frlsc_t);
        end
   else
        [e,frlsc]=max(results(R).frlsc,[],2); frlsc=C(frlsc);
   end
   cm=confusion(frlsc,DataStruct.GroundTruth(U)); results(R).cm_rlsc=cm; 
   results(R).err_rlsc=100*(1-sum(diag(cm))/sum(cm(:)));
        
fprintf(1,'split=%d LapRLS (transduction) err = %f \n',R, results(R).err_rlsc);
end

fprintf(1,'\n\n');
disp('LapRLS (transduction) mean confusion matrix');
disp(round(mean(reshape(vertcat(results.cm_rlsc)',[size(results(1).cm_rlsc,1) size(results(1).cm_rlsc,1) length(results)]),3)));
meanL = mean(vertcat(results.err_rlsc)); stdL = std(vertcat(results.err_rlsc));
fprintf(1,'LapRLS (transduction mean(std)) err = %f (%f)  \n',mean(vertcat(results.err_rlsc)),std(vertcat(results.err_rlsc)));
% if exist('Xt','var')
%     fprintf(1,'\n\n');
%     disp('LapRLS (out-of-sample) mean confusion matrix');
%     disp(round(mean(reshape(vertcat(results.cm_rlsc_t)',[size(results(1).cm_rlsc_t,1) size(results(1).cm_rlsc_t,1) length(results)]),3)));
% 
%     fprintf(1,'LapRLS (out-of-sample mean(std)) err = %f (%f)\n',mean(vertcat(results.err_rlsc_t)),std(vertcat(results.err_rlsc_t)));
% end
end


  
function [f1,b]=adjustbias(f,bias)
     jj=ceil((1-bias)*length(f));
     g=sort(f);
     b=0.5*(g(jj)+g(jj+1));
     f1=f-b;
end
     
function  index = foldslabel(x, rate)
labelnum = max(x);   %最大类别数
count = [];
%rand('seed', 1001);
globalind = 0;
for i = 1:labelnum
    eachnum = numel(find(x == i));
    randnum = sort(randi([globalind + 1, globalind + eachnum],floor(rate * eachnum), 1));
    count = [count; randnum];
    globalind = globalind+eachnum;
end
index = count;
end


function idxLabs = genlabidx(label, num)
totalclass = length(unique(label));
classnum = [];
for i = 1:totalclass
    classnum(i) = length(find(label == i));
end
idxLabs = [];
if num< min(classnum)
    for iclass = 1: totalclass
     tmp = find(label == iclass);
     idxLab1 = randsrc(1,num,tmp');
     idxLabs=[idxLabs,idxLab1];
    end
else
    num = min(classnum);
    for iclass = 1: totalclass
        tmp = find(label == iclass);
        idxLab1 = randsrc(1,num,tmp');
        idxLabs=[idxLabs,idxLab1];
    end
end


end
