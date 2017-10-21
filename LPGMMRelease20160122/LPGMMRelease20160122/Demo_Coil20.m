clear;clc;
%%% Load Data
load('COIL20.mat');      
gY = gnd; 
gX = NormalizeFea(double(fea)',0);

% load('CBCL3000.mat');      
% gY = label';  
% gX = NormalizeFea(double(test),0);

optLPGMM = []; optLPGMM.n_size = 4;     %%% Coil20 data
TtrainList = [20:5:70];

%% data preprocessing
[Dim,N] = size(gX);
Classlabels=unique(gY);
Classnum = length(Classlabels);


%% NtrainList = Range of Varying l/C
NtrainList = [1 2 4 6 8 10 12]; 
% NtrainList = [4 6 8 10 12 14 16]; 
%% Parameters for comparing algorithms
n_size = 6;

Nround = 10;                    %%% Repeat the experiment Nround times   20

AccLPGMM = [];    AccGGMC = [];   AccLGC = [];    AccGRF = []; AccKLP = []; AccFLAP = []; AccLapRLS = [];

for T = 1:Nround
    
    %% Random permutation of the data points
    rnperm = randperm(N);
    dataX = gX(:,rnperm);
    labelY = gY(rnperm);
    %% index of each class
    Dind=cell(Classnum,1);
    for iter1=1:Classnum,
        Dind{iter1}=find(labelY==Classlabels(iter1));
    end
    %%
    
    [Wlap] = DoubleWforMR(dataX, 'k', n_size);   %% Weight matrix
    
    %%% LapRLCS
    options = [];
    options.Kernel = 'linear';
    options.KernelParam = 1;
    options.NN = 10; options.k =2 ; options.GraphDistanceFunction = 'cosine'; options.GraphWeights = 'heat';
    options.GraphWeightParam = 'default'; options.GraphNormalize = 1; options.ClassEdges = 0;options.LaplacianDegree = 5;
    options.mu = 0.5;
    W = full(Wlap);
    D = sum(W(:,:),2);
    options.GraphNormalize = 1;options.LaplacianDegree = 5;
    if options.GraphNormalize==0
        L = spdiags(D,0,speye(size(W,1)))-W;
    else % normalized laplacian
        %             fprintf(1, 'Normalizing the Graph Laplacian\n');
        D(find(D))=sqrt(1./D(find(D)));
        D=spdiags(D,0,speye(size(W,1)));
        W=D*W*D;
        L=speye(size(W,1))-W;
    end
    options.gamma_A=0.01; options.gamma_I=0.1;
    
    K=Deform(options.gamma_I/options.gamma_A,calckernel(options,dataX'),L^options.LaplacianDegree);
    
    for iter1 = 1:length(NtrainList)
        %% index of labeled training: ind1
        ind1 =[]; 
        %% indenx of unlabeled training, which are also the test point(for SSL): ind2
        ind2=[];
        for c=1:Classnum  
            Ntrain = NtrainList(iter1);
            ind1 = [ind1; Dind{c}(1:Ntrain)];              %%%  training index
            ind2 = [ind2; Dind{c}((1+Ntrain):end)];        %%%  test index
        end
        dataTrain = dataX(:,ind1);    labelTrain = labelY(ind1);      %%% labeled training data
        dataTest  = dataX(:,ind2);    labelTest  = labelY(ind2);      %%% unlabeled traing Also the test data

        %% ---------LPGMM Classification--------

        AccLPGMM(T,iter1) = LPGMM(dataTrain, labelTrain, dataTest, labelTest,optLPGMM)
        
        %%% -------Learning with Local and Global Consistency
        graph=TransductionModel(Wlap);
        graph.prior=ones(1,Classnum);
        [predict, error]=lgc(dataX',labelY,ind1,graph);
        AccLGC(T,iter1) = 100*(1-error)
        %%
        
        %%% -------Learning with Local and Global Consistency graph transductive learning
        [predict, error]=grf(dataX',labelY,ind1,Wlap);
        AccGRF(T,iter1) = 100*(1-error)
        %%%
        
        %%% -------Greedy Gradient based Max-Cut algorithm GGMC
        [predict, error]=ggmc(dataX',labelY,ind1,graph);
        AccGGMC(T,iter1) = 100*(1-error)
        %%% -------
        
        %%% -------KLP
        DataStruct.data = dataX';
        DataStruct.GroundTruth = labelY;
        DataStruct.LabeledIndex = ind1;
        DataStruct.P = full(Wlap);
        option.korder = 4;
        [F,labelcount]=klp(DataStruct,option);
        [F_value,Classification] = max(labelcount,[],2);
        [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (labelY,Classification,Classlabels);
        AccKLP(T,iter1) = Accuracy * 100
        %%% -------
        
        %%% --- FLAP
        Alpha                  = 0.99;         % trade-off parameter
        yita                   = 0.1;          % eta
        [P, gamma] = ComputeP(full(Wlap), yita);
        Y = zeros(length(DataStruct.GroundTruth),Classnum);
        for i=1:size(DataStruct.LabeledIndex,1)
            ind = DataStruct.LabeledIndex(i);
            if(DataStruct.GroundTruth(ind,1) >= 1)
                Y(ind,DataStruct.GroundTruth(ind))=1;
            end
        end
        F = FLAPLabelPropagation(P, Y, Alpha);
        [~,Classification] = max(F,[],2);
        [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (DataStruct.GroundTruth,Classification,Classlabels);
        %display(['FLAP accuracy = ',num2str(Accuracy)]);
        AccFLAP(T,iter1) = Accuracy * 100
        
        %%% --- LapRLS
        C=unique(labelY);
        nclassifiers=length(C);
%         X = dataX';
%         Lind=ind1;
%         U=1:size(K,1);
%         U(Lind)=[];
%         data.K=K(Lind,Lind);
%         data.X=X(Lind,:); data.Y=labelY(Lind); % labeled data
%         %% % LapRLSC
%         classifier_rlsc=rlsc(options,data);
%         testdata.K=K(U,Lind');
% %         testdata.K = K(U,U);
%         [prbep_rlsc,f]=test_prbep(classifier_rlsc,testdata.K,labelY(U));
%        % disp(['Laplacian RLS Performance on split ' num2str(R) ': ' num2str(prbep_rlsc(R))]);
%        [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (DataStruct.GroundTruth(U),round(f),Classlabels);
%         AccLapRLS(T,iter1) = Accuracy * 100


    end
end


% iind =alphaList;
iind =NtrainList;

figure; hold on;

% errorbar(iind,mean(AccLPGMM,1), 0.1*std(AccLPGMM),...
%     'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','y','MarkerSize',8);%%%

errorbar(iind,mean(AccLGC,1), 0.1*std(AccLGC),...
    'm<-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% Traditional MR

errorbar(iind,mean(AccGRF,1), 0.1*std(AccGRF),...
    'bs-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% Traditional MR

errorbar(iind,mean(AccGGMC,1), 0.1*std(AccGGMC),...
    'b^-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% Traditional MR

errorbar(iind,mean(AccKLP,1), 0.1*std(AccKLP),...
    'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','y','MarkerSize',8) %%% Traditional MR

xlabel('Number of Labeled points (Varying l/C)','fontsize',10); ylabel('Accuracy (%)','fontsize',10);
grid on;
% legend('LPGMM', 'LGC','GRF','GGMC','KLP');
legend( 'LGC','GRF','GGMC','KLP');
hold off


pause;


%% Parameters for comparing algorithms
%  Ntrain = 2;
%  
%  n_size = 6;   
% 
% Nround = 10;                    %%% Repeat the experiment Nround times
% 
% AccLPGMM = [];    AccGGMC = [];   AccLGC = [];    AccGRF = []; AccKLP = [];


% for T = 1:Nround
%     
%     %% data preprocessing
%     [Dim,N2] = size(gX);
%     Classlabels=unique(gY);   
%     Classnum = length(Classlabels);            
%     rnperm = randperm(N2);     dataX1 = gX(:,rnperm);       labelY1 = gY(rnperm);    
%     
%     for iter2 = 1:length(TtrainList)
%         
%         Tnum = TtrainList(iter2);      ind3 =[];
%         for iter1=1:Classnum
%             indC = find(labelY1==Classlabels(iter1));
%             ind3 = [ind3; indC(1:Tnum)];                     %%% Total training index, where only 2 points are labeled !! Different from previous experiments
%         end
%         dataX = dataX1(:,ind3);  labelY = labelY1(ind3);%%% OverWrites these variables
%         
%         [Dim,N] = size(dataX);
%         %%
%         dataY = zeros(Classnum,N);   Dind=cell(Classnum,1);
%         for iter1=1:Classnum,
%             Dind{iter1}=find(labelY==Classlabels(iter1));                           
%         end
%         %%
%         
%         
%         [Wlap] = DoubleWforMR(dataX, 'k', n_size);   %% Weight matrix
%  
%         
%         ind1 =[]; ind2=[];
%         for iter1=1:Classnum
%             ind1 = [ind1; Dind{iter1}(1:Ntrain)];              %%% Only 2 points are labeled!!
%             ind2 = [ind2; Dind{iter1}((1+Ntrain):end)];        %%% unlabeled data index
%         end
%         dataTrain = dataX(:,ind1);  labelTrain = labelY(ind1);    %%% 
%         dataTest  = dataX(:,ind2);  labelTest  = labelY(ind2);    %%%
%         
%         %% ---------LPGMM Classification--------
% 
% %         AccLPGMM(T,iter2) = LPGMM(dataTrain, labelTrain, dataTest, labelTest,optLPGMM)
%         
%         
%         %% -------Learning with Local and Global Consistency
%         graph=TransductionModel(Wlap);
%         graph.prior=ones(1,Classnum);
%         [predict, error]=lgc(dataX',labelY,ind1,graph);
%         AccLGC(T,iter2) = 100*(1-error)
%         %%
%         
%         %% -------Learning with Local and Global Consistency graph transductive learning
%         [predict, error]=grf(dataX',labelY,ind1,Wlap);
%         AccGRF(T,iter2) = 100*(1-error)
%         %%
%         
%         %% -------ggmc
%         [predict, error]=ggmc(dataX',labelY,ind1,graph);
%         AccGGMC(T,iter2) = 100*(1-error)
%         %% -------
% 
%          %%% -------KLP
%         DataStruct.data = dataX';
%         DataStruct.GroundTruth = labelY;
%         DataStruct.LabeledIndex = ind1;
%         DataStruct.P = full(Wlap);
%         option.korder = 4;
%         [F,labelcount]=klp(DataStruct,option);
%         [F_value,Classification] = max(labelcount,[],2);
%         [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (labelY,Classification,Classlabels);
%          
%         AccKLP(T,iter2) = Accuracy * 100
%     end
% end


% iind =alphaList;
% iind =TtrainList;
% 
% figure; hold on;

% errorbar(iind,mean(AccLPGMM,1), 0.1*std(AccLPGMM),...
%     'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','y','MarkerSize',8);%%%

% errorbar(iind,mean(AccLGC,1), 0.1*std(AccLGC),...
%     'm<-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%%  
% 
% errorbar(iind,mean(AccGRF,1), 0.1*std(AccGRF),...
%     'bs-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% 
% 
% errorbar(iind,mean(AccGGMC,1), 0.1*std(AccGGMC),...
%     'b^-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% 
% 
% errorbar(iind,mean(AccKLP,1), 0.1*std(AccKLP),...
%     'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% 
% 
% xlabel('Number of Training points (Varying N/C)','fontsize',10); ylabel('Accuracy (%)','fontsize',10);
% grid on;
% legend('LGC','GRF','GGMC','KLP');
% hold off








