clear all
close all;
clc;
korder                 = 4;            % higher order paramter
% filename               = 'COIL20.mat';
% filename               = 'ORL.mat';
%              filename = 'TDT2.mat';
filename = 'breast.mat';
% filename = 'wine.mat';
options = [];
options.Metric = 'Euclidean';
%                    options.Metric = 'Cosine';
options.NeighborMode = 'KNN';
options.WeightMode = 'HeatKernel';
%             options.WeightMode = 'Cosine';

%       options.Adaptive = 'Yes';
options.Adaptive = 'No';
options.korder = korder;
for K = [ 10, 15, 20]
    for sigma = [0.5,1,2]
        for labelrate = [0.05,0.10, 0.15, 0.2, 0.25,0.3]
            
            %% Parameters
            % sigma                  = 0.5;          % Kernel width
            % K                      = 10;           % Nearest neighbors
            % % alpha                  = 0.7;         % trade-off parameter
            % % yita                   = 0.1;          % eta
            
            % labelrate              =0.15;
            
            options.k = K;
            options.sigma = sigma;
            for ai=1
                %% Load Dataset
                % load Iris.mat;
                % Data = Iris.Data;
                % GroundTruth = Iris.GroundTruth;
                % LabeledIndex = Iris.LabeledIndex;
                % UnlabeledIndex = Iris.UnlabeledIndex;
                % 0.1 0.96838
                % 0.05 0.95392
                % 0.15 accuracy = 0.97041
                % 0.20 accuracy = 0.97158
                %0.25 accuracy = 0.97147
                
                % DataStruct = loaddata('TDT2.mat',labelrate);
                
                %0.1 accuracy = 0.84097
                %0.1 accuracy = 0.84167
                %0.15 accuracy = 0.83681
                % 0.2 accuracy = 0.85833
                % 0.25 0.85486
            end
            fid = fopen('result1.txt','a');
            fprintf(fid, 'File = %s\r\n',filename );
            fprintf(fid, 'Metric = %s\r\n',options.Metric );
            fprintf(fid, 'WeightMode = %s\r\n',options.WeightMode );
            fprintf(fid, 'K size = %s\r\n',num2str(options.k));
            fprintf(fid, 'label rate = %s\r\n',num2str(labelrate) );
            fprintf(fid, 'korder = %s\r\n',num2str(korder) );
            fprintf(fid, 'sigma = %s\r\n',num2str(sigma) );
            AccSAFER=[];AccFLAP=[];AccGGMC=[]; AccKLP=[]; AccHARM=[]; AccLGC=[];AccLPGMM = [];AccGRF = [];
            for iter = 1:20
                %% construct data
                DataStruct = loaddata(filename,labelrate);
                
                
                %%  construct Graph
                W = constructW(DataStruct.data,options);
                
                P = full(W);
                DataStruct.P = P;
                GroundTruth = DataStruct.GroundTruth;
                % ClassTotal  = max(GroundTruth);            % number of classes
                ClassTotal  = length(unique(GroundTruth));
                classes = (1:ClassTotal)';
                %% % SAFER
                %             predict = SAFER_f(DataStruct,options.k);
                %             [v, Classification] = max(predict, [],2);
                prediction1 = Self_KNN(DataStruct.data(DataStruct.LabeledIndex),DataStruct.data(DataStruct.UnlabeledIndex),DataStruct.GroundTruth(DataStruct.LabeledIndex),'euclidean',K);
                prediction2 = Self_KNN(DataStruct.data(DataStruct.LabeledIndex),DataStruct.data(DataStruct.UnlabeledIndex),DataStruct.GroundTruth(DataStruct.LabeledIndex),'cosine',K);
                candidate_prediction = [prediction1 prediction2];
                
                baseline_prediction = KNN(DataStruct.data(DataStruct.LabeledIndex),DataStruct.data(DataStruct.UnlabeledIndex), GroundTruth);
                
                [Safer_prediction]= SAFER(candidate_prediction,baseline_prediction);
                
                [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (DataStruct.GroundTruth(DataStruct.UnlabeledIndex),Safer_prediction,classes);
                %display(['SAFER accuracy = ',num2str(Accuracy)]);
                AccSAFER(iter) = Accuracy*100;
                %% harmonic function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                W_ll = P(DataStruct.LabeledIndex, DataStruct.LabeledIndex);
                W_lu = P(DataStruct.LabeledIndex,DataStruct.UnlabeledIndex);
                W_ul = P(DataStruct.UnlabeledIndex,DataStruct.LabeledIndex);
                W_uu = P(DataStruct.UnlabeledIndex, DataStruct.UnlabeledIndex);
                W_arrage = [[W_ll, W_lu]; [W_ul, W_uu]];
                f_l = zeros(size(DataStruct.LabeledIndex , 1), ClassTotal);
                for i=1:size(DataStruct.LabeledIndex,1)
                    ind = DataStruct.LabeledIndex(i);
                    if(GroundTruth(ind,1) >= 1)
                        f_l(i,GroundTruth(ind))=1;
                    end
                end
                [fu, fu_CMN] = harmonic_function(W_arrage,f_l);
                
                [v, Classification] = max(fu_CMN, [],2);
                
                [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth(DataStruct.UnlabeledIndex),Classification,classes);
                %                 display(['Harmonic accuracy = ',num2str(Accuracy)]);
                AccHARM(iter) = Accuracy*100;
                %% FLAP  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                Alpha                  = 0.99;         % trade-off parameter
                yita                   = 0.1;          % eta
                [P, gamma] = ComputeP(W, yita);
                Y = zeros(length(DataStruct.GroundTruth),ClassTotal);
                for i=1:size(DataStruct.LabeledIndex,1)
                    ind = DataStruct.LabeledIndex(i);
                    if(GroundTruth(ind,1) >= 1)
                        Y(ind,GroundTruth(ind))=1;
                    end
                end
                F = FLAPLabelPropagation(P, Y, Alpha);
                [~,Classification] = max(F,[],2);
                [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (DataStruct.GroundTruth(DataStruct.UnlabeledIndex),Classification(DataStruct.UnlabeledIndex),classes);
                %display(['FLAP accuracy = ',num2str(Accuracy)]);
                AccFLAP(iter) = Accuracy*100;
                for i=1
                    % num0 = 100; X = twomoon_gen(num0); c = 2; y = [ones(num0,1);2*ones(num0,1)];
                    % LabeledIndex = DataStruct.LabeledIndex;
                    % UnlabeledIndex = DataStruct.UnlabeledIndex;
                    % Data = DataStruct.data;
                    % [DataTotal,Dim] = size(Data);
                    %% Construct graph
                    % W = ConstructKNNGraph(X, sigma, 5);
                    % % [P, gamma] = ComputeP(W, yita);
                    % P = W;
                    %%  第一种构造图
                    %       options = [];
                    %       options.Metric = 'Euclidean'; %0.96
                    % %       options.Metric = 'Cosine';
                    %       options.NeighborMode = 'KNN';
                    %       options.k = K;
                    %       options.WeightMode = 'HeatKernel';
                    % % options.WeightMode = 'Cosine';
                    %       options.t = sigma;
                    % %       options.Adaptive = 'Yes';
                    % options.Adaptive = 'No';
                    %
                    %       W = constructW(Data,options);
                    %
                    %        P = full(W);
                    %%  第二种构造图
                    %%%W=LLE_Graph(X,K,Metric)
                    %%% Input:
                    %%%       --- X:         Input Dataset (nXd)
                    %%%       --- K:         Number of the nearest neighbors
                    %%%       --- Metric:   'Euclidean' - use Euclidean distance evaluate the similarity between samples;
                    %%%                     'Cosine'    - use cosine distance to evaluate the similarity between samples.
                    
                    %%
                    %      options = [];  % 1
                    %       options.Metric = 'Euclidean';
                    %       options.NeighborMode = 'Supervised';
                    %       options.gnd = GroundTruth;
                    %       options.bLDA = 1;
                    %       W = constructW(Data,options);
                    %%
                    % options = [];
                    %       options.Metric = 'Euclidean';
                    %       options.NeighborMode = 'Supervised';
                    %       options.gnd = GroundTruth;
                    %       options.WeightMode = 'HeatKernel';
                    %       options.t = 1;
                    %       W = constructW(Data,options);
                    %       P = full(W);
                    %%
                    % metric =  'Euclidean'; % 0.91
                    % metric =  'Cosine'; %0.96
                    % W = LLE_Graph(Data,K ,metric);
                    % P = full(W);
                    %  figure;
                    %  imshow(P,[]); colormap jet; colorbar;hold on;
                    %  figure;
                    %  for i = 1:num0
                    %      if(y(i) == 1)
                    %          plot(X(i,1),X(i,2),'.r');
                    %      else
                    %          plot(X(i,1),X(i,2),'.b');
                    %      end;
                    %  end;
                    %  hold on;
                    %  plotadj(W,X(:,1),X(:,2));
                    %  scatter3(Data);
                    % plotadj(W,Data,Data);
                    % imshow(A0,[]); colormap jet; colorbar;
                    % hold on;
                    %% Label Propagation
                    % Y = zeros(DataTotal,ClassTotal);
                    % %for i=1:DataTotal
                    % for i=1:size(LabeledIndex,1)
                    %     ind = LabeledIndex(i);
                    %     %Y(i,GroundTruth(i,1))=1;
                    %     if(GroundTruth(ind,1) >= 1)
                    %         Y(ind,GroundTruth(ind))=1;
                    %     end
                    % end
                    % % Y(UnlabeledIndex,:) = 0;
                    % % l = size(LabeledIndex,1);
                    % % n = DataTotal;
                    % % % l = size(fl, 1); % the number of labeled points
                    % % % n = size(P, 1); % total number of points
                    % %
                    % % % the graph Laplacian L=D-W
                    % % L = diag(sum(P)) - P;
                    % %
                    % % fl=Y(LabeledIndex,ClassTotal);
                    % % % the harmonic function.
                    % % %  fu = - inv(L(l+1:n, l+1:n)) * L(l+1:n, 1:l) * fl; %%%Matrix is close to singular or badly scaled.
                    % %
                    % % fu = - pinv(eye(n-l,n-l)-L(l+1:n, l+1:n)) * L(l+1:n, 1:l) * fl;
                    % %
                    % % % compute the CMN solution
                    % % q = sum(fl)+1; % the unnormalized class proportion estimate from labeled data, with Laplace smoothing
                    % % fu_CMN = fu .* repmat(q./sum(fu), n-l, 1);
                    % % Classification = fu;
                    % e = 0.000001;       % stopping criterion
                    % f_threshold = 0.001;
                    % %label count 阈值
                    % l_threshold = 3/4*korder;
                    % t = 1;
                    %
                    % F=zeros(DataTotal,ClassTotal,korder);
                    % for i = 1:korder   %init
                    %     F(:,:,i) = Y;
                    % end
                    %
                    % F_last = zeros(size(F));
                    % A_k = k_order(P,DataTotal,korder);
                    %
                    % maxlabel = zeros(DataTotal,ClassTotal ,korder);  % 将F中每一行最大值位置在maxlabel中赋为1
                    %
                    % while 1
                    %    for k = 1: korder  % 通过k阶传播
                    %        F(:,:,k) = alpha * A_k(:,:,k) * F(:,:,k) + (1-alpha)*Y;
                    %    end
                    %     if sumnorm(F-F_last,'fro')<e || t==200  %停止条件
                    %         break;
                    %     else
                    %
                    % %         for i=1:DataTotal   % 小于阈值的标签分数赋值为0
                    % %             for j =1:ClassTotal
                    % %                 if F(i,j) < f_threshold
                    % %                     F(i,j) = 0;
                    % %                 end
                    % %             end
                    % %         end
                    %
                    %         %投票筛选， 在K阶转移矩阵中通过投票方式选定标签出现次数最多的标签
                    %         % 找到当前每阶的F行最大值， 并以下标为label
                    %         for k1 = 1:korder
                    %             for nsamp = 1: DataTotal
                    %                 [~, ind]=max(F(nsamp, :, k1));
                    %                 maxlabel(nsamp, ind, k1) =1 ;
                    %             end
                    %         end
                    %         %vote 找到出现在每阶函数中出现次数最多的标签
                    %         labelcount = sum(maxlabel, 3);
                    %         %[~, tepind] = max(labelcount, [], 2);
                    %         %修改F矩阵
                    %         for fi = 1: DataTotal
                    %             for fj = 1: ClassTotal
                    %                 if(labelcount(fi, fj) >= l_threshold)
                    %                     F(fi,:,:) = zeros(ClassTotal,korder);
                    %                     F(fi,fj,:) = ones(1,korder);
                    %                 end
                    %             end
                    %         end
                    %          % 归一化
                    %          for norm1 = 1: korder
                    %             for i =1:DataTotal
                    %                 sum0 = sum(F(i,:,norm1));
                    %                 if sum0 >0
                    %                     for j = 1:ClassTotal
                    %                         F(i,j,norm1) = F(i,j,norm1)/sum0;
                    %                     end
                    %                 end
                    %             end
                    %          end
                    %         %进行下一轮判断
                    %         t = t + 1;
                    %         F_last = F;
                    %     end
                    % end
                end
                
                %%  LGC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                LGCClassification = Mylgc(DataStruct);
                [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth(DataStruct.UnlabeledIndex),LGCClassification(DataStruct.UnlabeledIndex),classes);
                %display(['LGC accuracy = ',num2str(Accuracy)]);
                AccLGC(iter) = Accuracy*100;
                
                %% klp%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                F = klp(DataStruct,options);
                labelcount = sum(F, 3);
                % [F_value,Classification] = max(F,[],2);  % Classification = maxvalue_index
                [F_value,Classification] = max(labelcount,[],2);
                [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth(DataStruct.UnlabeledIndex),Classification(DataStruct.UnlabeledIndex),classes);
                
                %display(['KLP accuracy = ',num2str(Accuracy)]);
                AccKLP(iter) = Accuracy*100;
                
                %%% LPGMM
                optLPGMM = []; optLPGMM.n_size = K;
                [~,LPGMMidx] = LPGMM(DataStruct.data(DataStruct.LabeledIndex)',DataStruct.GroundTruth(DataStruct.LabeledIndex),DataStruct.data(DataStruct.UnlabeledIndex)',DataStruct.GroundTruth(DataStruct.UnlabeledIndex),optLPGMM);
                [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth(DataStruct.UnlabeledIndex),LPGMMidx,classes);
                AccLPGMM(iter) = Accuracy*100;
                
                graph=TransductionModel(P);
                graph.prior=ones(1,ClassTotal);
                %%% -------Learning with Local and Global Consistency graph transductive learning
                [predict, error]=grf(DataStruct.data',DataStruct.GroundTruth,DataStruct.LabeledIndex,P);
                AccGRF(iter) = 100*(1-error);
                %%% -------Greedy Gradient based Max-Cut algorithm GGMC
                [predict, error]=ggmc(DataStruct.data',DataStruct.GroundTruth,DataStruct.LabeledIndex,graph);
                AccGGMC(iter) = 100*(1-error);
                
                %
                % fprintf(fid, 'File = %s\r\n',filename );
                % % fprintf(fid, 'File = %s\r\n',options.Metric );
                % fprintf(fid, 'rate = %s\r\n',num2str(labelrate) );
                % fprintf(fid, 'korder = %s\r\n',num2str(korder) );
                % fprintf(fid, 'Accuracy = %s\r\n',num2str(Accuracy) );
                % % fprintf(fid, 'precision = %s\r\n',num2str(precision) );
                % % fprintf(fid, 'numcorrect = %s\r\n',num2str(numcorrect) );
                % fprintf(fid, 'recall = %s\r\n',num2str(recall) );
                % fprintf(fid, 'F = %s\r\n',num2str(F) );
                % % fprintf(fid, 'PatN = %s\r\n',num2str(PatN) );
                % fprintf(fid, 'MAP = %s\r\n',num2str(MAP) );
                % fprintf(fid, 'sigma = %s\r\n',num2str(sigma) );
                % fprintf(fid, 'K= %s\r\n',num2str(K) );
                
            end
            fprintf(fid, 'LPGMM = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccLPGMM,2)),num2str( 0.1*std(AccLPGMM)));
            fprintf(fid, 'FLAP = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccFLAP,2)),num2str( 0.1*std(AccFLAP)));
            fprintf(fid, 'LGC = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccLGC,2)),num2str( 0.1*std(AccLGC)));
            fprintf(fid, 'harmonic function = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccHARM,2)),num2str( 0.1*std(AccHARM)));
            fprintf(fid, 'GRF = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccGRF,2)),num2str( 0.1*std(AccGRF)));
            fprintf(fid, 'GGMC = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccGGMC,2)),num2str( 0.1*std(AccGGMC)));
            fprintf(fid, 'SAFER = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccSAFER,2)),num2str( 0.1*std(AccSAFER)));
            fprintf(fid, 'KLP = mean Accuracy: %s, std = %s \r\n',num2str(mean(AccKLP,2)),num2str( 0.1*std(AccKLP)));
            fclose(fid);
        end
    end
end






