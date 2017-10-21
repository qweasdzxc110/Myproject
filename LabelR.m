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
k=repmat(sum(A,2),[1,length(A)]); % �Ⱦ���
P0=A./k; % ���Զ�
Ppre=A*P0; %��ʼ����
a=1;
%COM={};
count=0;
%%
   % Step2: Inflation ʹ�ô�ĸ��ʸ���С�ĸ�С
   while a
       Pnow=A*Ppre;
       Pin=Pnow.^in ;
       k=repmat(sum(Pin,2),[1,length(A)]);
       Pnow=Pin./k;
 %%
   % Step3: Cutoff ����С����ֵʱ�� �趨Ϊ0
       index= Pnow<r;
       Pnow(index)=0;
%%
   % Step4: Explicit Conditional Update ����ǰ������ֵ����������ƶȴ���q����ôֹͣ���¸õ�
    MaNow=max(Pnow,[],2);
    MaPre=max(Ppre,[],2);
    restart=[];
       for i=1:length(A)
           gain=0;
           Nb=find( Aori(i,:));  %����
           MaxI=max(Pnow(i,:));  %��ǰ�������
           MaxI=find(Pnow(i,:)==MaxI); %�±�
           MaxNb=MaNow(Nb);   %�õ����ڼ�����������
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
        Pnow(restart,:)=Ppre(restart,:);  %ֹͣ����
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