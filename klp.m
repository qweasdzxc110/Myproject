function [F,labelcount] = klp(DataStruct,options)
% sigma                  = 0.5;          % Kernel width
% K                      = 10;           % Nearest neighbors
% korder                 = 4;            % higher order paramter
alpha                  = 0.7;         % trade-off parameter

korder = options.korder;
GroundTruth = DataStruct.GroundTruth;
LabeledIndex = DataStruct.LabeledIndex;
% UnlabeledIndex = DataStruct.UnlabeledIndex;
Data = DataStruct.data;
[DataTotal,Dim] = size(Data);
% [DataTotal,Dim] = size(DataStruct.P);
ClassTotal  = max(GroundTruth);            % number of classes

for tempor=1
% Goptions = [];
% 
% Goptions.k = options.k;
% Goptions.Metric = options.Metric;
% Goptions.NeighborMode = options.NeighborMode;
% Goptions.WeightMode = options.WeightMode;
% Goptions.t = options.sigma;
% Goptions.Adaptive = options.Adaptive;


% num0 = 100; X = twomoon_gen(num0); c = 2; y = [ones(num0,1);2*ones(num0,1)];
%% Construct graph
% W = ConstructKNNGraph(X, sigma, 5);
% % [P, gamma] = ComputeP(W, yita);
% P = W;
%%  第一种构造图 
     

%       W = constructW(Data,Goptions);
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
end
%% Label Propagation
Y = zeros(DataTotal,ClassTotal);
%for i=1:DataTotal
for i=1:size(LabeledIndex,1)
    ind = LabeledIndex(i);
    %Y(i,GroundTruth(i,1))=1;
    if(GroundTruth(ind,1) >= 1)
        Y(ind,GroundTruth(ind))=1;
    end 
end
%% 
for tempor =1
% Y(UnlabeledIndex,:) = 0;
% l = size(LabeledIndex,1);
% n = DataTotal;
% % l = size(fl, 1); % the number of labeled points
% % n = size(P, 1); % total number of points
% 
% % the graph Laplacian L=D-W
% L = diag(sum(P)) - P;
% 
% fl=Y(LabeledIndex,ClassTotal);
% % the harmonic function.
% %  fu = - inv(L(l+1:n, l+1:n)) * L(l+1:n, 1:l) * fl; %%%Matrix is close to singular or badly scaled.
% 
% fu = - pinv(eye(n-l,n-l)-L(l+1:n, l+1:n)) * L(l+1:n, 1:l) * fl;
% 
% % compute the CMN solution
% q = sum(fl)+1; % the unnormalized class proportion estimate from labeled data, with Laplace smoothing
% fu_CMN = fu .* repmat(q./sum(fu), n-l, 1);
% Classification = fu;
end
%%
e = 0.000001;       % stopping criterion
f_threshold = 0.0001;
%label count 阈值
l_threshold = 3/4*korder;
% l_threshold = korder;
t = 1;
%P = DataStruct.P;
% P = w(i,j)/ sum(w(i,k))
% T = zeros(size(DataStruct.P,1),size(DataStruct.P,2),korder);
A_k = k_order(DataStruct.P,DataTotal,korder);
T = A_k;
% for k1 = 1:korder
% for s1 = 1:size(DataStruct.P,1)
%     sumT =  sum(T(s1,:,k1), 2);
%     for s2 = 1:size(DataStruct.P,1)
% %         if(T(s1,s2,k1) > 0)
%             T(s1,s2,k1) = T(s1,s2,k1)/sumT;
% %         end
%     end
% end
% end
% for s1 = 1:size(DataStruct.P,1)
%     sumP =  sum(DataStruct.P(s1,:), 2);
%     for s2 = 1:size(DataStruct.P,1)
%         if(DataStruct.P(s1,s2) > 0)
%             P(s1,s2) = DataStruct.P(s1,s2)/sumP;
%         end
%     end
% end
F=zeros(DataTotal,ClassTotal,korder);
% F_last = zeros(DataTotal,ClassTotal,korder);
PseLabel = zeros(DataTotal, ClassTotal);
for i = 1:korder   %init
    F(:,:,i) = Y;
end


% A_k = k_order(P,DataTotal,korder);
%  labelcount = zeros(DataTotal,ClassTotal);


while 1
   F_last = F;
   for k = 1: korder  % 通过k阶传播
%        F(:,:,k) = alpha * A_k(:,:,k) * F(:,:,k) + (1-alpha)*Y;
%        F(:,:,k) =  A_k(:,:,k) * F(:,:,k);
         F(:,:,k) = alpha * T(:,:,k) * F(:,:,k) + (1-alpha)*Y;
   end
 
%         for i=1:DataTotal   % 小于阈值的标签分数赋值为0
%             for j =1:ClassTotal
%                 if F(i,j) < f_threshold
%                     F(i,j) = 0; 
%                 end
%             end
%         end
       
        %投票筛选， 在K阶转移矩阵中通过投票方式选定标签出现次数最多的标签
        % 找到当前每阶的F行最大值， 并以下标为label
        maxlabel = zeros(DataTotal,ClassTotal ,korder);  % 将F中每一行最大值位置在maxlabel中赋为1
        for k1 = 1:korder
            for nsamp = 1: DataTotal
                [~, ind]=max(F(nsamp, :, k1));
                maxlabel(nsamp, ind, k1) =1 ;
            end
        end
        %vote 找到出现在每阶函数中出现次数最多的标签
       
        labelcount = sum(maxlabel, 3);
        %[~, tepind] = max(labelcount, [], 2);
        %修改F矩阵 ——》 伪标签矩阵
        for fi = 1: DataTotal
            for fj = 1: ClassTotal
                if(labelcount(fi, fj) >= l_threshold)
%                     F(fi,:,:) = zeros(ClassTotal,korder);
%                     F(fi,fj,:) = ones(1,korder);
                      PseLabel(fi,fj) = 1;
                end
            end
        end
         % 归一化
         for norm1 = 1: korder
            for i =1:DataTotal
                sum0 = sum(F(i,:,norm1));
                if sum0 >0
                    for j = 1:ClassTotal
                        F(i,j,norm1) = F(i,j,norm1)/sum0;
                    end
                end
            end
         end 
    
       % 融合PseLabel和真实标签Y
       [YPserow,YPsecol] = find(PseLabel == 1);
       for YPi = 1:size(YPserow,1)
           if(Y(YPserow(YPi),:) == 0)
               Y(YPserow(YPi), YPsecol(YPi)) = 1;
           end
       end
       
       % fix orginal label matrix
%        for kind = 1:korder
%         for i=1:size(LabeledIndex,1)
%              ind = LabeledIndex(i);
%              if(GroundTruth(ind,1) >= 1)
%                  F(ind,:,kind) = zeros(1,ClassTotal);
%                  F(ind,GroundTruth(ind),kind)=1;
%              end 
%         end
%        end
        for kind = 1:korder
            [Yrow,Ycol] = find(Y == 1);
            for ri = 1:size(Yrow,1)
                F(Yrow(ri),:,kind) = 0;
                F(Yrow(ri),Ycol(ri),kind) = 1;
            end
        end
      %进行下一轮判断
        t = t + 1;
        
    if sumnorm(F-F_last,'fro')<e || t==200  %停止条件
        break;
    end    
       
end
end