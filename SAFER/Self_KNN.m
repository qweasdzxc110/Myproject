function Self_KNN_prediction = Self_KNN(label_instance, unlabel_instance, label, distance_measure, k)

    label_l = label;
	[idx, dist] = knnsearch(label_instance,unlabel_instance,'dist',distance_measure,'k',k);
    %formal
    %label_u = mean(label(idx),2);
    label_u = max(label(idx),1);
    for i =1:length(label_u) 
        a = tabulate(label_u(i,:));
        [val,ind] = max(a(:,2));
        la(i) = a(ind,1);
    end
    instance = [label_instance;unlabel_instance];
%     label = [label_l;label_u];
    label = [label_l;la'];
    [idx, dist] = knnsearch(instance,unlabel_instance,'dist',distance_measure,'k',k);
    for i = 1 : 5
        %label_u = mean(label(idx),2);
        %         label_u = max(label(idx),2);
        %label_last = label;
        label_u1 = max(label(idx),1);
        for j =1:length(label_u1)
            a = tabulate(label_u1(j,:));
            %             [~,la1(j)] = max(a(:,2));
            [val,ind1] = max(a(:,2));
            la1(j) = a(ind1,1);
        end
        label = [label_l;la1'];
    end
    for j =1:length(label_u1)
        a = tabulate(label_u1(j,:));
        %[~,la2(j)] = max(a(:,2));
        [val,ind2] = max(a(:,2));
        la2(j) = a(ind2,1);
    end
    Self_KNN_prediction = la2';
end
