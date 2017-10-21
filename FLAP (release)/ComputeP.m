
function [P, gamma] = ComputeP(W, yita)

%% compute gamma
P1 = W.*W;
gamma = yita*(1/max(sum(P1,2)));

%% compute matrix P
P = gamma*P1;
P = P + diag(1-sum(P,2));


