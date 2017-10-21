%%%% Experiments for the JMLR paper (fig.4 g&h&i)
%%% by Jun Wang(jwang@ee.columbia.edu)

clear;
clc;
close all;
path(path,'GraphConstruct');
load datasets/noisytwomoon.mat;

X=data;
data_num=length(label);
knn=6;
sig=cal_sigma(X,knn);
Test_num=50;
Imbalance_num=[1 2 4 6 8 10 12 14 16 18 20];
gnd=label;
gnd(label==-1)=2;
N=size(X,1);
KK=X*X'; 
DD=diag(KK)*ones(1,N)+ones(N,1)*diag(KK)'-2*KK; 
DD=sqrt(DD);


%%% KNN graph
fprintf('KNN graph constructing ... \n');
connectionoptions.KernelType='Linear';    
connectionoptions.Display=0;
connectionoptions.KB=knn;    
connectionoptions.Model='KNN';
[P, Pnn]=ConnectionModel(X,connectionoptions);
%sig=cal_sigma_neighbor(Pnn,DD,10);   
    

for type_j=1:3
    weightoptions.Type=type_j;
    weightoptions.KB=knn;
    weightoptions.Display=0;
    weightoptions.KernelSize=sig;
    W=WeightingModel(X,P,Pnn,weightoptions);
    W=full(W);
    graph=TransductionModel(W);
    graph.prior=[1 1];
    fprintf('####################KNN Graph K=%d Weight Model=%d#################### \n', knn, type_j);
    
    for imbalance_i=1:length(Imbalance_num)
        fprintf('The imbalance ratio: %d \n', imbalance_i);
        for test_j=1:Test_num
            num_positive=1;
            ind_positive=find(label==1);
            rand('seed',(imbalance_i-1)*Test_num+test_j);
            rand_ind_positive=randperm(length(ind_positive));
            labeled_ind_positive=ind_positive(rand_ind_positive(1));

            num_negative=1;
            ind_negative=find(label==-1);
            rand('seed',(imbalance_i-1)*Test_num+test_j);
            rand_ind_negative=randperm(length(ind_negative));
            labeled_ind_negative=ind_negative(rand_ind_negative(1:Imbalance_num(imbalance_i)));

            labeled_ind=[labeled_ind_positive labeled_ind_negative'];
            unlabeled_ind=setdiff([1:data_num],labeled_ind);



            [predict, error]=grf(X,gnd,labeled_ind,W);
            knn_grf_error(imbalance_i,type_j,test_j)=error;

            [predict, error]=lgc(X,gnd,labeled_ind,graph);
            knn_lgc_error(imbalance_i,type_j,test_j)=error;

            [predict, error]=ggmc(X,gnd,labeled_ind,graph);
            knn_ggmc_error(imbalance_i,type_j,test_j)=error;
            
            
            fprintf('Round %d   LGC: %f   HFGF: %f   GGMC: %f\n',test_j,knn_lgc_error(imbalance_i,type_j,test_j),knn_grf_error(imbalance_i,type_j,test_j),knn_ggmc_error(imbalance_i,type_j,test_j));
            
         end
        fprintf('####################Summay --- KNN Graph K=%d Weight Model=%d Number of labels=%d #################### \n',knn,type_j,imbalance_i);
        fprintf('LGC-Mean: %f   HFGF-Mean: %f   GGMC-Mean: %f\n',mean(knn_lgc_error(imbalance_i,type_j,:)),mean(knn_grf_error(imbalance_i,type_j,:)),mean(knn_ggmc_error(imbalance_i,type_j,:)));
    end
end
save noisytwomoon_knn_imbalancenum_results.mat;

fig=figure; hold on;
X=Imbalance_num;
plot(X,mean(reshape(knn_lgc_error(:,1,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,1,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,1,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('Imbalance Ratio');ylabel('Error Rate');
grid on;box on;


fig=figure; hold on;
X=Imbalance_num;
plot(X,mean(reshape(knn_lgc_error(:,2,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,2,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,2,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('Imbalance Ratio');
ylabel('Error Rate');
grid on;box on;

fig=figure; hold on;
X=Imbalance_num;
plot(X,mean(reshape(knn_lgc_error(:,3,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,3,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,3,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('Imbalance Ratio');ylabel('Error Rate');
grid on;box on;

