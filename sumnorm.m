
function SumNorm = sumnorm(X1,metric)
[row, col,height ] = size(X1);
SumNorm = 0;
if strcmp(metric, 'fro')
    for i = 1:height
        SumNorm = SumNorm + norm(X1(:,:,i), metric);
    end
end
end 