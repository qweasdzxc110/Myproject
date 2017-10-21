function [R,count]=LabelR(A,in,r,q)
% 
%  
% Inputs :
% A : adjacent matrix
% in : Inflation parameter
%   : default =2
% q : Conditional Update parameter
%        default = 0.7
% r   : Cut off parameter
%  : default = 0.1
% Output :
% R : label classfication
%%
   %  Step1 : Propagation
Aori=A;            
A=A+eye(length(A));%  Aii = 1
k=repmat(sum(A,2),[1,length(A)]); % 度矩阵
P0=A./k; % 除以度
Ppre=A*P0; %初始传播
a=1;
%COM={};
count=0;
%%
   % Step2: Inflation 使得大的概率更大，小的更小
   while a
       Pnow=A*Ppre;
       Pin=Pnow.^in ;
       k=repmat(sum(Pin,2),[1,length(A)]);
       Pnow=Pin./k;
 %%
   % Step3: Cutoff 概率小于阈值时， 设定为0
       index= Pnow<r;
       Pnow(index)=0;
%%
   % Step4: Explicit Conditional Update 若当前最大概率值和其近邻相似度大于q，那么停止更新该点
    MaNow=max(Pnow,[],2);
    MaPre=max(Ppre,[],2);
    restart=[];
       for i=1:length(A)
           gain=0;
           Nb=find( Aori(i,:));  %近邻
           MaxI=max(Pnow(i,:));  %当前概率最大
           MaxI=find(Pnow(i,:)==MaxI); %下标
           MaxNb=MaNow(Nb);   %得到近邻集合中最大概率
           for k=1:length(Nb)
               MaxNbID=find(Pnow(Nb(k),:)==MaxNb(k));
               if all(ismember(MaxI,MaxNbID));
                   gain=gain+1;
               end
           end
           if gain>=q*length(Nb)
              restart=[i,restart];
           end
       end
        Pnow(restart,:)=Ppre(restart,:);  %停止更新
 %%
   % Step5: Stop Criterion
       if all(ismember(find(Pnow(i,:)==MaNow(i)),find(Ppre(i,:)==MaPre(i))))
           a=0;
       end
       Ppre=Pnow;
       count=count+1;
   end
   R=Pnow;
end