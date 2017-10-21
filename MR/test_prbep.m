function [prbep,f]=test_prbep(classifier,K,Y);
   f=K(:,classifier.svs)*classifier.alpha-classifier.b;
   [m,n,maxcm]=classifier_evaluation(f,Y,'pre_rec_equal');
   prbep=100*(maxcm(1,1)/(maxcm(1,1)+maxcm(1,2)));
end

