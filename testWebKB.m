%% Multi view dataset: Link.mat, paga.mat, linkpage.mat, binary classification


load 'LINK.mat'; 
options.NN = 10;
% options.GraphDistanceFunction = 'euclidean';
Wtotal = zeros(size(X,1),size(X,1), 3);
% [L1, Wtotal(:,:,1), options1]=laplacian(options,X);
[L1, W1, options1]=laplacian(options,X);

load 'PAGE.mat';
options.NN = 10;
% options.GraphDistanceFunction = 'euclidean';
% [L2, Wtotal(:,:,2), options2] = laplacian(options, X);
[L2, W2, options2] = laplacian(options, X);

load 'PAGELINK.mat';
options.NN = 10;
% options.GraphDistanceFunction = 'euclidean';
% [L3,Wtotal(:,:,3), options3] = laplacian(options, X);
[L3,W3, options3] = laplacian(options, X);

options.gamma_A=0.01; options.gamma_I=0.1;
L=(L1+L2+L3)/3; 
W = full((W1+W2+W3)/3);
clear L1 L2  L3;
% clear W1 W2  W3;

mode = 'page+link';

switch mode

case 'link'
    load LINK.mat;
case 'page'
    load PAGE.mat
case 'page+link'
    load PAGELINK.mat
end

GroundTruth = zeros(size(X,1),1);
    for j = 1:size(X,1)
        if(Y(j) == 1)
            GroundTruth(j) = 1;
        else
            GroundTruth(j) = 2;
        end
    end
 classes = (1:2)';

K=Deform(options.gamma_I/options.gamma_A,calckernel(options,X),L^options.LaplacianDegree);
predictLGC = zeros(size(X,1),size(idxLabs,1));
DataStruct = [];
for R=1:size(idxLabs,1)
	L=idxLabs(R,:);
    U=1:size(K,1); 
    U(L)=[];
	data.K=K(L,L); 
    data.X=X(L,:); data.Y=Y(L); % labeled data
    
    % % LapRLSC
    classifier_rlsc=rlsc(options,data);
    testdata.K=K(U,L);
    prbep_rlsc(R)=test_prbep(classifier_rlsc,testdata.K,Y(U));
    disp(['Laplacian RLS Performance on split ' num2str(R) ': ' num2str(prbep_rlsc(R))]);
    
    %%  LGC 
    alpha=0.95;
%     predict = zeros(size(X,1),3);
%     for wi = 1:3
%         W = Wtotal(:,:,wi);
    
    
    D=full(diag(sum(W)));
%     D1=D^(-0.5);
    D1 = pinv(D^(0.5));
%     D1 = D^(0.5)\ones(length(W));
    S=D1*W*D1;
    I=eye(length(W),length(W));
    IS=(I-alpha*S)^(-1);
    Yl= zeros(size(X,1),length(unique(Y)));
    for i=1:length(L)
        if( data.Y(i)== 1)
            Yl(L(i),1) = 1;
        else
            Yl(L(i),2) = 1;
        end
    end
    F=IS*Yl;
    [a, b]=max(F,[],2);
%     predict(:,wi)=b; 
    predictLGC(:,R)=b; 
    [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth,predictLGC(:,R),classes);
    Accuracy
    
    
    %% KLP
optionsKLP.korder = 3;
DataStruct.GroundTruth =  GroundTruth;
DataStruct.LabeledIndex = L';
DataStruct.data = X;
DataStruct.P = W;
[F,labelcount] = klp(DataStruct,optionsKLP);
% labelcount = sum(F, 3);
% [F_value,Classification] = max(F,[],2);  % Classification = maxvalue_index
[F_value,Classification] = max(labelcount,[],2);
[confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth,Classification,classes);

display(['KLP accuracy = ',num2str(Accuracy)]);
end
%     % vote
%     a = [1,1,2];
%     b=unique(a);
%     c=histc(a,b);  c =2     1
%     finalpred = zeros(size(X,1), 1);
%     
%     for i  = 1: size(X,1 )
%         b = unique(predict(i,:));
%         c = histc(predict(i, :), b);
%         [maxv, maxi] = max(c);
%         finalpred(i) = maxi;
%     end
%     [confus,Accuracy,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (GroundTruth,finalpred,classes)





