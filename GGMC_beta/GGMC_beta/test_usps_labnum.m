%%%% Experiments for the JMLR paper (fig.5)
%%%Jun Wang, Tony Jebara, Shih-Fu Chang. Semi-Supervised Learning Using Greedy Max-Cut. Journal of Machine Learning Research, 14:729-758, March 2013
%%% by Jun Wang(jwang@ee.columbia.edu)

clear;
clc;
close all;
path(path,'GraphConstruct');

load datasets/USPS_alldigits.mat;
%%% random select 4K for exp
rand('seed',2010);
rand_ind=randperm(length(label));
X=X(rand_ind(1:4000),:);
label=label(rand_ind(1:4000));
clear rand_ind;

data_num=length(label);
knn=6;
sig=cal_sigma(X,knn);
Test_num=20;
Labels_num=[20 30 40 50 60 70 80 90 100];

N=size(X,1);
KK=X*X'; 
DD=diag(KK)*ones(1,N)+ones(N,1)*diag(KK)'-2*KK; 
DD=sqrt(DD);
class_num=length(unique(label));

%%% KNN graph
fprintf('KNN graph constructing ... \n');
connectionoptions.KernelType='Linear';    
connectionoptions.Display=0;
connectionoptions.KB=knn;    
connectionoptions.Model='KNN';
[P, Pnn]=ConnectionModel(X,connectionoptions);
  

gnd=label;

for type_j=1:3
    weightoptions.Type=type_j;
    weightoptions.KB=knn;
    weightoptions.Display=0;
    weightoptions.KernelSize=sig;
    W=WeightingModel(X,P,Pnn,weightoptions);
    W=full(W);
    graph=TransductionModel(W);
    graph.prior=ones(1,class_num);

    fprintf('####################KNN Graph K=%d Weight Model=%d#################### \n', knn, type_j);
    
    for labels_i=1:length(Labels_num)
        labnum=Labels_num(labels_i);
        fprintf('The number of labels: %d \n', labnum);
        for test_j=1:Test_num
            ct=cputime;
            labeled_ind=[];
            for ii=1:class_num
                ind_class=find(label==ii);
                rand('seed',(labels_i-1)*Test_num+(test_j-1)*class_num+ii);
                rand_ind=randperm(length(ind_class));
                
                labeled_ind=[labeled_ind ind_class(rand_ind(1))];
                unlabeled_ind=setdiff([1:data_num],labeled_ind);                
            end

            if labnum>class_num
                rand('seed',(labels_i-1)*Test_num+test_j);
                candidate_labels=setdiff([1:data_num],labeled_ind);
                rand_ind_more=randperm(length(candidate_labels));
                labeled_ind=[labeled_ind candidate_labels(rand_ind_more(1:labnum-class_num))];
                unlabeled_ind=setdiff([1:data_num],labeled_ind);
            end


            [predict, error]=grf(X,gnd,labeled_ind,W);
            knn_grf_error(labels_i,type_j,test_j)=error;

            [predict, error]=lgc(X,gnd,labeled_ind,graph);
            knn_lgc_error(labels_i,type_j,test_j)=error;

            [predict, error]=ggmc(X,gnd,labeled_ind,graph);
            knn_ggmc_error(labels_i,type_j,test_j)=error;

            
            fprintf('Round %d   LGC: %f   HFGF: %f   GGMC: %f\n',test_j,knn_lgc_error(labels_i,type_j,test_j),knn_grf_error(labels_i,type_j,test_j),knn_ggmc_error(labels_i,type_j,test_j));
            disp(cputime-ct);
         end
        fprintf('####################Summay --- KNN Graph K=%d Weight Model=%d Number of labels=%d #################### \n',knn,type_j,labnum);
        fprintf('LGC-Mean: %f   HFGF-Mean: %f   GGMC-Mean: %f\n',mean(knn_lgc_error(labels_i,type_j,:)),mean(knn_grf_error(labels_i,type_j,:)),mean(knn_ggmc_error(labels_i,type_j,:)));

    end
end

fig=figure; hold on;
X=Labels_num;
plot(X,mean(reshape(knn_lgc_error(:,1,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,1,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,1,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('The number of labels');
ylabel('Error Rate');
grid on;box on;


fig=figure; hold on;
X=Labels_num;
plot(X,mean(reshape(knn_lgc_error(:,2,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,2,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,2,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('The number of labelds');
ylabel('Error Rate');
grid on;box on;

fig=figure; hold on;
X=Labels_num;
plot(X,mean(reshape(knn_lgc_error(:,3,:),size(knn_lgc_error,1),size(knn_lgc_error,3))'),'-bd','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_grf_error(:,3,:),size(knn_grf_error,1),size(knn_grf_error,3))'),'-gh','LineWidth',6,'MarkerSize',30);
plot(X,mean(reshape(knn_ggmc_error(:,3,:),size(knn_ggmc_error,1),size(knn_ggmc_error,3))'),'-ro','LineWidth',6,'MarkerSize',30);
legend('LGC','HFGF','GGMC');
xlabel('The number of labels');
ylabel('Error Rate');
grid on;box on;


