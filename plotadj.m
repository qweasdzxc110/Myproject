function plotadj(A,x,y)
DEF=length(A);
plot(x,y,'ro');      
for i=1:DEF 
    clear a;
    a=find(A(i,:)>0);%��A����ÿ�д���0�������ڸ��еĵ�ַ�ҳ�������a�� 
    if ~isempty(a)
        for j=1:length(a)
          %  c=num2str(A(i,a(j))); %��A�е�Ȩֵת��Ϊ�ַ���    
           % text((x(i)+x(a(j)))/2,(y(i)+y(a(j)))/2,c,'Fontsize',10);%��Ȩֵ��ʾ�����������м�   
           % hold on; 
            line([x(i) x(a(j))],[y(i) y(a(j))]);%���� 
        end  
    end
end 