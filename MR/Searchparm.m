
dataset = 'glass.mat';
for sigma = 1: 10
    for degree = 1:10
        [~, meanL, stdL] = MyLapRLSC(dataset, 10, sigma, degree);
        fprintf('mean:%f, std: %f',meanL,  stdL);
    end
end