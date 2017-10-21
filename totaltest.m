clear;clc;
%%% Load Data
filename = 'COIL20.mat';
load(filename);      
gY = gnd; 
gX = NormalizeFea(double(fea)',0);

% load('CBCL3000.mat');      
% gY = label';  
% gX = NormalizeFea(double(test),0);

optLPGMM = []; optLPGMM.n_size = 4;     %%% Coil20 data
%TtrainList = [20:5:70];

%% data preprocessing
[Dim,N] = size(gX);
Classlabels=unique(gY);
Classnum = length(Classlabels);


%% NtrainList = Range of Varying l/C
%NtrainList = [1 2 4 6 8 10 12 14 16]; 
NtrainList = [1 2]; 
% NtrainList = [4 6 8 10 12 14 16]; 

%% Parameters for comparing algorithms
n_size = 6;

Nround = 1;                    %%% Repeat the experiment Nround times   20

%AccLPGMM = [];    AccGGMC = [];   AccLGC = [];    AccGRF = []; AccKLP = []; AccFLAP = []; AccLapRLS = [];
AccuracyLPGMM=zeros(Nround,length(NtrainList)); AccuracyGGMC=[]; AccuracyLGC=[]; AccuracyGRF=[]; AccuracyKLP=[]; AccuracyFLAP=[]; 
precisionLPGMM=[]; precisionGGMC=[]; precisionLGC=[]; precisionGRF=[]; precisionKLP=[]; precisionFLAP=[];
recallLPGMM =[]; recallGGMC =[]; recallLGC =[]; recallGRF =[]; recallKLP =[]; recallFLAP =[]; 

MAPLPGMM = [];MAPGGMC = [];MAPLGC = []; MAPGRF = [];MAPKLP = []; MAPFLAP = [];

% for n_size = [ 10, 15, 20]  
%for n_size = [ 10]
for K = [ 10]
for sigma = [0.5,1,2]
%for labelrate = [0.05,0.10, 0.15, 0.2, 0.25,0.3]
    fid = fopen('result.txt','a');
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
    options = [];
      options.Metric = 'Euclidean'; 
%       options.Metric = 'Cosine';
      options.NeighborMode = 'KNN';
      options.k = K;
      options.WeightMode = 'HeatKernel';
% options.WeightMode = 'Cosine';
      options.sigma = sigma;
%       options.Adaptive = 'Yes';
      options.Adaptive = 'No';
    W = constructW(dataX',options);
    Wlap = full(W);
   %[Wlap] = DoubleWforMR(dataX, 'k', K, sigma);   %% Weight matrix
    
%     %%% LapRLCS
%     options = [];
%     options.Kernel = 'linear';
%     options.KernelParam = sigma;
%     options.NN = n_size; %options.k =2 ; 
%     options.GraphDistanceFunction = 'cosine'; options.GraphWeights = 'heat';
%     options.GraphWeightParam = 'default'; options.GraphNormalize = 1; options.ClassEdges = 0;options.LaplacianDegree = 5;
%     options.mu = 0.5;
%     W = full(Wlap);
%     D = sum(W(:,:),2);
%     options.GraphNormalize = 1;options.LaplacianDegree = 5;
%     if options.GraphNormalize==0
%         L = spdiags(D,0,speye(size(W,1)))-W;
%     else % normalized laplacian
%         %             fprintf(1, 'Normalizing the Graph Laplacian\n');
%         D(find(D))=sqrt(1./D(find(D)));
%         D=spdiags(D,0,speye(size(W,1)));
%         W=D*W*D;
%         L=speye(size(W,1))-W;
%     end
%     options.gamma_A=0.01; options.gamma_I=0.1;
    
    %K=Deform(options.gamma_I/options.gamma_A,calckernel(options,dataX'),L^options.LaplacianDegree);
    
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

        %AccLPGMM(T,iter1) = LPGMM(dataTrain, labelTrain, dataTest, labelTest,optLPGMM)
       % [AccLPGMM(T,iter1), ydata]= LPGMM(dataTrain, labelTrain, dataTest, labelTest,optLPGMM);
        %[confus,AccuracyLPGMM(T,iter1),numcorrect,precisionLPGMM(T,iter1),recallLPGMM(T,iter1),FLPGMM(T,iter1),PatN,MAPLPGMM(T,iter1),NDCGatN] = compute_accuracy_F (labelY,ydata,Classlabels);
        
        %AccLPGMM(T,iter1) = LPGMMver2(dataTrain, labelTrain, dataTest, labelTest,optLPGMM)
        
        %%% -------Learning with Local and Global Consistency
        graph=TransductionModel(Wlap);
        graph.prior=ones(1,Classnum);
        [predict, error]=lgc(dataX',labelY,ind1,graph);
        %[confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (labelY,predict',Classlabels);
        
        [confus,AccuracyLGC(T,iter1),numcorrect,precision,recall,F,PatN,MAPLGC(T,iter1),NDCGatN] = compute_accuracy_F (labelY(ind2),predict(ind2)',Classlabels);
        %AccLGC(T,iter1) = 100*(1-error)
        %%
        
        %%% -------Learning with Local and Global Consistency graph transductive learning
        [predict, error]=grf(dataX',labelY,ind1,Wlap);
        [confus,AccuracyGRF(T,iter1),numcorrect,precision,recall,F,PatN,MAPGRF(T,iter1),NDCGatN] = compute_accuracy_F (labelY(ind2),predict',Classlabels);
        %AccGRF(T,iter1) = 100*(1-error)
        %%%
        
        %%% -------Greedy Gradient based Max-Cut algorithm GGMC
        [predict, error]=ggmc(dataX',labelY,ind1,graph);
        [confus,AccuracyGGMC(T, iter1),numcorrect,precision,recall,F,PatN,MAPGGMC(T, iter1),NDCGatN] = compute_accuracy_F (labelY(ind2),predict(ind2)',Classlabels);
        %AccGGMC(T,iter1) = 100*(1-error)
        %%% -------
        
        %%% -------KLP
        DataStruct.data = dataX';
        DataStruct.GroundTruth = labelY;
        DataStruct.LabeledIndex = ind1;
        DataStruct.UnlabeledIndex = ind2;
        DataStruct.P = full(Wlap);
        option.korder = 4;
        [F,labelcount]=klp(DataStruct,option);
        [F_value,Classification] = max(labelcount,[],2);
        [confus,AccuracyKLP(T,iter1),numcorrect,precision,recall,F,PatN,MAPKLP(T,iter1),NDCGatN] = compute_accuracy_F (labelY(ind2),Classification(ind2),Classlabels);
        %AccKLP(T,iter1) = Accuracy * 100
        %%% -------
        pred = SAFER_T(DataStruct,K);
        
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
        [confus,AccuracyFLAP(T,iter1),numcorrect,precision,recall,F,PatN,MAPFLAP(T,iter1),NDCGatN] = compute_accuracy_F (labelY(ind2),Classification(ind2),Classlabels);
        %display(['FLAP accuracy = ',num2str(Accuracy)]);
        %AccFLAP(T,iter1) = Accuracy * 100
        
       
        
        
        %%% --- LapRLS
%         C=unique(labelY);
%         nclassifiers=length(C);
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
        fprintf(fid, 'File = %s\r\n',filename );
        % fprintf(fid, 'File = %s\r\n',options.Metric );
        fprintf(fid, 'labelrate = %s\r\n',num2str(labelrate) );
        fprintf(fid, 'korder = %s\r\n',num2str(option.korder) );
        fprintf(fid, 'k SIZE = %s\r\n',num2str(n_size) );
        
        fprintf(fid, 'FLAP mean Accuracy = %s,  std = %s\r\n',num2str(mean(AccuracyFLAP*100)), num2str( 0.1*std(AccuracyFLAP)));
        fprintf(fid, 'LGC mean Accuracy = %s,  std = %s\r\n',num2str(mean(AccuracyLGC*100)), num2str( 0.1*std(AccuracyLGC)));
        fprintf(fid, 'GRF mean Accuracy = %s,  std = %s\r\n',num2str(mean(AccuracyGRF*100)), num2str( 0.1*std(AccuracyGRF)));
        fprintf(fid, 'LPGMM mean Accuracy = %s,  std = %s\r\n',num2str(mean(AccuracyLPGMM*100)), num2str( 0.1*std(AccuracyLPGMM)));
        fprintf(fid, 'GGMC mean Accuracy = %s,  std = %s\r\n',num2str(mean(AccuracyGGMC*100)), num2str(0.1* std(AccuracyGGMC)));
        fprintf(fid, 'KLP mean Accuracy = %s,  std = %s\r\n',num2str(mean(AccuracyKLP*100)), num2str( 0.1*std(AccuracyKLP)));
        
        fprintf(fid, 'FLAP mean MAP = %s,  std = %s\r\n',num2str(mean(MAPFLAP)), num2str( std(MAPFLAP)));
        fprintf(fid, 'LGC mean MAP = %s,  std = %s\r\n',num2str(mean(MAPLGC)), num2str( std(MAPLGC)));
        fprintf(fid, 'GRF mean MAP = %s,  std = %s\r\n',num2str(mean(MAPGRF)), num2str( std(MAPGRF)));
        fprintf(fid, 'LPGMM mean MAP = %s,  std = %s\r\n',num2str(mean(MAPLPGMM)), num2str( std(MAPLPGMM)));
        fprintf(fid, 'GGMC mean MAP = %s,  std = %s\r\n',num2str(mean(MAPGGMC)), num2str( std(MAPGGMC)));
        fprintf(fid, 'KLP mean MAP = %s,  std = %s\r\n',num2str(mean(MAPKLP)), num2str( std(MAPKLP)));
        
        % fprintf(fid, 'precision = %s\r\n',num2str(precision) );
%         % fprintf(fid, 'numcorrect = %s\r\n',num2str(numcorrect) );
%         fprintf(fid, 'recall = %s\r\n',num2str(recall) );
%         fprintf(fid, 'F = %s\r\n',num2str(F) );
%         % fprintf(fid, 'PatN = %s\r\n',num2str(PatN) );
%         fprintf(fid, 'MAP = %s\r\n',num2str(MAP) );
%         fprintf(fid, 'sigma = %s\r\n',num2str(sigma) );
%         fprintf(fid, 'K= %s\r\n',num2str(K) );
        fclose(fid);
end
end



for i =1
% % iind =alphaList;
% iind =NtrainList;
% 
% figure; hold on;
% 
% % errorbar(iind,mean(AccLPGMM,1), 0.1*std(AccLPGMM),...
% %     'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','y','MarkerSize',8);%%%
% 
% errorbar(iind,mean(AccLGC,1), 0.1*std(AccLGC),...
%     'm<-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% Traditional MR
% 
% errorbar(iind,mean(AccGRF,1), 0.1*std(AccGRF),...
%     'bs-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% Traditional MR
% 
% errorbar(iind,mean(AccGGMC,1), 0.1*std(AccGGMC),...
%     'b^-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','g','MarkerSize',8) %%% Traditional MR
% 
% errorbar(iind,mean(AccKLP,1), 0.1*std(AccKLP),...
%     'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','y','MarkerSize',8) %%% Traditional MR
% 
% xlabel('Number of Labeled points (Varying l/C)','fontsize',10); ylabel('Accuracy (%)','fontsize',10);
% grid on;
% % legend('LPGMM', 'LGC','GRF','GGMC','KLP');
% legend( 'LGC','GRF','GGMC','KLP');
% hold off
% 
% 
% pause;


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
end