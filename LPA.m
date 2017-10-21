%LPA算法的思路：
%
%      首先每个节点有一个自己特有的标签，节点会选择自己邻居中出现次数最多的标签，如果每个标签出现次数一样多，那么就随机选择一个标签替换自己原始的标签，如此往复，直到每个节点标签不再发生变化，那么持有相同标签的节点就归为一个社区。
%
%算法优点：思路简单，时间复杂度低，适合大型复杂网络。
%
%算法缺点：众所周知，划分结果不稳定，随机性强是这个算法致命的缺点。
%
%                 体现在：（1）更新顺序。节点标签更新顺序随机，但是很明显，越重要的节点越早更新会加速收敛过程
%
%                             （2）随机选择。如果一个节点的出现次数最大的邻居标签不止一个时，随机选择一个标签作为自己标签。很明显，在标签重复次数相同的情况下，与本节点相似度更高或对本节点影响力越大的邻居节点的标签有更大的概率被节点选中


function [ Labelnew ] = LPA( adjacent_matrix,label )
    if nargin<2
        label = 1:size(adjacent_matrix,2);
    end
    N = size(adjacent_matrix,2);
    
    Label1 = label;
    Label2 = Label1;
    Labelnew = Label1;
    flag=1;
    while(1)
        for i=1:N
            nb_lables = Labelnew(adjacent_matrix(i,:)==1);%找到邻居下标对应的标签
            if size(nb_lables,2)>0
                x = tabulate(nb_lables);
                max_nb_labels = x(x(:,2)==max(x(:,2)),1);
                Labelnew(i) = max_nb_labels(randi(length(max_nb_labels)));
            end
        end
        % 收敛条件,预防跳跃
        if all(Labelnew==Label1)||all(Labelnew==Label2)
            break;
        else
            if flag==1
                Label1 = Labelnew;
                flag=0;
            else
                Label2 = Labelnew;
                flag=1;
            end
        end
    end
end