
function w=ConstructKNNGraph(data, sigma, k)
% data number * dimension 
% sigma :kernel width 
% k : the number of nearest neighbor

sample_total=size(data,1);

%find distance matrix（sample_total*sample_total）
distance_matrix = dist_mat(data,data);
distance_matrix = distance_matrix + diag(inf(1,sample_total));




%find NN matrix
% neighbor_matrix=zeros(sample_total,k);
% for i=1:sample_total
%     [~,index]=sort(distance_matrix(i,:),'ascend');
%     neighbor_matrix(i,:)=index(1:k);
% end
% %%%
%%
neighbor_matrix=zeros(sample_total,k);
neighbor_matrix_data=zeros(sample_total,sample_total);
geo_matrix = zeros(sample_total,sample_total);
for i=1:sample_total
    [~,index]=sort(distance_matrix(i,:),'ascend');
    %% 增加二阶相似矩阵 
    
    
    neighbor_matrix_data(i,index(1:k))=distance_matrix(i,index(1:k));
    neighbor_matrix(i,:)=index(1:k);
end
ug = sparse(neighbor_matrix_data);
for i = 1:sample_total
    for j = 1:sample_total
        if(i ~= j)
% scatter(data(:,1),data(:,2));
            geo_matrix(i,j) = graphshortestpath(ug,i,j);
        end;
    end;
end;
% UG = sparse(data,data,neighbor_matrix);
% dist = graphshortestpath(UG,1,sample_total);
%%
%compute W
w = zeros(sample_total,sample_total);

for i = 1:sample_total
    for n = 1:k
%         Euclidean_distance = distance_matrix(i,neighbor_matrix(i,n));
%         w(i,neighbor_matrix(i,n)) = exp(-Euclidean_distance/(2*sigma^2));
        w(i,neighbor_matrix(i,n)) = exp(-geo_matrix(i,n)/(2*sigma^2));
        w(neighbor_matrix(i,n), i) = w(i,neighbor_matrix(i,n));
    end
end

w=(w+w')/2;
end

function D=dist_mat(P1, P2)
%
% Euclidian distances between vectors
P1 = double(P1);
P2 = double(P2);

X1=repmat(sum(P1.^2,2),[1 size(P2,1)]);
X2=repmat(sum(P2.^2,2),[1 size(P1,1)]);
R=P1*P2';
D=sqrt(X1+X2'-2*R);
end


