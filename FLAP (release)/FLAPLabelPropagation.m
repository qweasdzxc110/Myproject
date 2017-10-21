%% FLAP propagation

function F = FLAPLabelPropagation(P, Y, alpha)

e = 0.0001;       % stopping criterion

t = 1;
F = Y; F_last = zeros(size(F));
while 1
    F = alpha * P * F + (1-alpha)*Y;
    if norm(F-F_last,'fro')<e || t==200
        break;
    else
        t = t + 1;
        F_last = F;
    end
end

 
