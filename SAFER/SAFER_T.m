function pred=SAFER_T(Data,k)
%load('breast.mat');
%data = loaddata('breast.mat',0.1);
%data = loaddata('COIL20.mat',0.1);
%data = loaddata('Yale.mat',0.1);
% data = loaddata('wine.mat', 0.1);
% data = loaddata('TDT2.mat', 0.1);
 
%label_num = size(data.LabeledIndex,1);
%unlabel_num = size(data.UnlabeledIndex,1);
X = minmax_normalized(Data.data);

label_instance = X(Data.GroundTruth(Data.LabeledIndex),:);
unlabel_instance = X(Data.GroundTruth(Data.UnlabeledIndex),:);
Ylabel = Data.GroundTruth(Data.LabeledIndex);

Uground_truth = Data.GroundTruth(Data.UnlabeledIndex);
prediction1 = Self_KNN(label_instance,unlabel_instance,Ylabel,'euclidean',k);
% pred1A = ceil(prediction1);
% pred1B = floor(prediction1);
prediction2 = Self_KNN(label_instance,unlabel_instance,Ylabel,'cosine',k);
% pred2A = ceil(prediction2);
% pred2B = floor(prediction2);

candidate_prediction = [prediction1 prediction2];
% candidate_predictionA = [pred1A pred2A];
% candidate_predictionB = [pred1B pred2B];
baseline_prediction = KNN(label_instance,unlabel_instance, Data.GroundTruth);

[Safer_prediction]= SAFER(candidate_prediction,baseline_prediction);
% predA = ceil(Safer_prediction);
% predB = floor(Safer_prediction);
Safer_predictionA = Safer_prediction;
Safer_predictionB = Safer_prediction;
for i = 1:length(Safer_prediction)
    if(Safer_prediction(i) - floor(Safer_prediction(i)) > 0.7)
        %Safer_prediction(i) = floor(Safer_prediction(i)) + 1;
        Safer_predictionA(i) = floor(Safer_prediction(i))+1;
        Safer_predictionB(i) = floor(Safer_prediction(i))+1;
    end
    if(Safer_prediction(i) - floor(Safer_prediction(i)) < 0.3)
        Safer_predictionA(i) = floor(Safer_prediction(i));
        Safer_predictionB(i) = floor(Safer_prediction(i));
    end
    if(Safer_prediction(i) - floor(Safer_prediction(i)) > 0.3 && Safer_prediction(i) - floor(Safer_prediction(i)) < 0.7)
        Safer_predictionA(i) = floor(Safer_prediction(i));
        Safer_predictionB(i) = floor(Safer_prediction(i)) +1 ;
    end
end
Classlabels = (1:length(unique(Data.GroundTruth)))';
% [confus,AccuracyA,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (Uground_truth,predA,Classlabels);
% [confus,AccuracyB,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (Uground_truth,predB,Classlabels);
[confus,AccuracyA,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (Uground_truth,Safer_predictionA,Classlabels);
 [confus,AccuracyB,numcorrect,precision,recall,F,PatN,MAP,NDCGatN] = compute_accuracy_F (Uground_truth,Safer_predictionB,Classlabels);
% mse = sum((Safer_prediction-Uground_truth).^2)/(label_num+unlabel_num);
if AccuracyA > AccuracyB
    pred = Safer_predictionA;
end
if AccuracyB >= AccuracyA
    pred = Safer_predictionB;
end
end
