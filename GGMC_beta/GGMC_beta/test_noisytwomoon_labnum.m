%%%% Experiments for the JMLR paper (fig.4 a&b&c)
%%%Jun Wang, Tony Jebara, Shih-Fu Chang. Semi-Supervised Learning Using Greedy Max-Cut. Journal of Machine Learning Research, 14:729-758, March 2013
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
Test_num=100;
Labels_num=10;
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
    
    for labels_i=2:Labels_num
        fprintf('The number of labels: %d \n', labels_i);
        for test_j=1:Test_num
            num_positive=1;
            ind_positive=find(label==1);
            rand('seed',(labels_i-1)*Test_num+test_j);
            rand_ind_positive=randperm(length(ind_positive));
            labeled_ind_positive=ind_positive(rand_ind_positive(1));

            num_negative=1;
            ind_negative=find(label==-1);
            rand('seed',(labels_i-1)*Test_num+test_j);
            rand_ind_negative=randperm(length(ind_negative));
            labeled_ind_negative=ind_negative(rand_ind_negative(1));

            labeled_ind=[labeled_ind_positive labeled_ind_negative];
            unlabeled_ind=setdiff([1:data_num],labeled_ind);

            if labels_i>2
                rand('seed',(labels_i-1)*Test_num+test_j);
                candidate_labels=setdiff([ind_positive' ind_negative'],labeled_ind);
                rand_ind_more=randperm(length(candidate_labels));
                labeled_ind=[labeled_ind candidate_labels(rand_ind_more(1:labels_i-2))];
                unlabeled_ind=setdiff([1:data_num],labeled_ind);
            end


            [predict, error]=grf(X,gnd,labeled_ind,W);
            knn_grf_error(labels_i-1,type_j,test_j)=error;

            [predict, error]=lgc(X,gnd,labeled_ind,graph);
            knn_lgc_error(labels_i-1,type_j,test_j)=error;

            [predict, error]=ggmc(X,gnd,labeled_ind,graph);
            knn_ggmc_error(labels_i-1,type_j,test_j)=error;
            

            fprintf('Round %d   LGC: %f   HFGF: %f   GGMC: %f\n',test_j,knn_lgc_error(labels_i-1,type_j,test_j),knn_grf_error(labels_i-1,type_j,test_j),knn_ggmc_error(labels_i-1,type_j,test_j));
            
         end
        fprintf('####################Summay --- KNN Graph K=%d Weight Model=%d Number of labels=%d #################### \n',knn,type_j,labels_i);
        fprintf('LGC-Mean: %f   HFGF-Mean: %f   LapSVM-Mean: %f   LapRLSC-Mean: %f   GGMC-Mean: %f\n',mean(knn_lgc_error(labels_i-1,type_j,:)),mean(knn_grf_error(labels_i-1,type_j,:)),mean(knn_ggmc_error(labels_i-1,type_j,:)));
    end
end


fig=figure; hold on;
X=[2:Labels_num];
plot(X,mean(reshape(knn_lgc_error(:,1,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,1,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,1,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('The number of labels');
ylabel('Error Rate');
grid on;box on;


fig=figure; hold on;
X=[2:Labels_num];
plot(X,mean(reshape(knn_lgc_error(:,2,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,2,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,2,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('The number of labels');
ylabel('Error Rate');
grid on;box on;

fig=figure; hold on;
X=[2:Labels_num];
plot(X,mean(reshape(knn_lgc_error(:,3,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,3,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,3,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('The number of labels');
ylabel('Error Rate');
grid on;box on;

