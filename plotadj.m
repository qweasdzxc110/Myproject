function plotadj(A,x,y)
DEF=length(A);
plot(x,y,'ro');      
for i=1:DEF 
    clear a;
    a=find(A(i,:)>0);%将A矩阵每行大于0的数的在该行的地址找出来放在a中 
    if ~isempty(a)
        for j=1:length(a)
          %  c=num2str(A(i,a(j))); %将A中的权值转化为字符型    
           % text((x(i)+x(a(j)))/2,(y(i)+y(a(j)))/2,c,'Fontsize',10);%将权值显示在两点连线中间   
           % hold on; 
            line([x(i) x(a(j))],[y(i) y(a(j))]);%连线 
        end  
    end
end 