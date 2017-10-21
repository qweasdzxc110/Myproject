[xa, ya, xxa, yya] = loadBinaryUSPS(0, 1);
labels_a=[ya' yya'];
labels_a(labels_a==1)=10;
labels_a(labels_a==-1)=1;

[xb, yb, xxb, yyb] = loadBinaryUSPS(2, 3);
labels_b=[yb' yyb'];
labels_b(labels_b==1)=2;
labels_b(labels_b==-1)=3;

[xc, yc, xxc, yyc] = loadBinaryUSPS(4, 5);
labels_c=[yc' yyc'];
labels_c(labels_c==1)=4;
labels_c(labels_c==-1)=5;

[xd, yd, xxd, yyd] = loadBinaryUSPS(6, 7);
labels_d=[yd' yyd'];
labels_d(labels_d==1)=6;
labels_d(labels_d==-1)=7;

[xe, ye, xxe, yye] = loadBinaryUSPS(8, 9);
labels_e=[ye' yye'];
labels_e(labels_e==1)=8;
labels_e(labels_e==-1)=9;


labels=[labels_a labels_b labels_c labels_d labels_e];
X=[xa; xxa; xb; xxb; xc; xxc; xd; xxd; xe; xxe;];