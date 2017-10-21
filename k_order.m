function A_k =k_order(A, num_nodes, step)
A = ScaleSimMat(A);                        
A_k = zeros(num_nodes, num_nodes, step);
A_k(:,:,1) = A;

for i = 2:step
    A_k(:,:,i) = A_k(:,:,i-1)*A;            %A_k(:,:,i) indicates A^k
end

end

function W = ScaleSimMat(W)

%scale 
W = W - diag(diag(W));  %diagonal elements must be 0
D = diag(sum(W), 0);    %degree matrix

% W = D^(-1)*W;
W = pinv(D)*W;
end

