%LPA�㷨��˼·��
%
%      ����ÿ���ڵ���һ���Լ����еı�ǩ���ڵ��ѡ���Լ��ھ��г��ִ������ı�ǩ�����ÿ����ǩ���ִ���һ���࣬��ô�����ѡ��һ����ǩ�滻�Լ�ԭʼ�ı�ǩ�����������ֱ��ÿ���ڵ��ǩ���ٷ����仯����ô������ͬ��ǩ�Ľڵ�͹�Ϊһ��������
%
%�㷨�ŵ㣺˼·�򵥣�ʱ�临�Ӷȵͣ��ʺϴ��͸������硣
%
%�㷨ȱ�㣺������֪�����ֽ�����ȶ��������ǿ������㷨������ȱ�㡣
%
%                 �����ڣ���1������˳�򡣽ڵ��ǩ����˳����������Ǻ����ԣ�Խ��Ҫ�Ľڵ�Խ����»������������
%
%                             ��2�����ѡ�����һ���ڵ�ĳ��ִ��������ھӱ�ǩ��ֹһ��ʱ�����ѡ��һ����ǩ��Ϊ�Լ���ǩ�������ԣ��ڱ�ǩ�ظ�������ͬ������£��뱾�ڵ����ƶȸ��߻�Ա��ڵ�Ӱ����Խ����ھӽڵ�ı�ǩ�и���ĸ��ʱ��ڵ�ѡ��


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
            nb_lables = Labelnew(adjacent_matrix(i,:)==1);%�ҵ��ھ��±��Ӧ�ı�ǩ
            if size(nb_lables,2)>0
                x = tabulate(nb_lables);
                max_nb_labels = x(x(:,2)==max(x(:,2)),1);
                Labelnew(i) = max_nb_labels(randi(length(max_nb_labels)));
            end
        end
        % ��������,Ԥ����Ծ
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