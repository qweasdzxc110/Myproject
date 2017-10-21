function Data = loaddata(datafile, labelrate)
    Data = [];
    if strcmp(datafile, 'TDT2.mat')
        load 'TDT2.mat';
        Data.data = tfidf(fea);
        Data.GroundTruth = gnd;
        [row, col] = size(gnd);
        Data.LabeledIndex = f(gnd, labelrate);
        Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
    end
    if strcmp(datafile, 'COIL20.mat')
        load 'COIL20.mat';
        fea = NormalizeFea(fea);
        Data.data = fea;
        Data.GroundTruth = gnd;
        [row, col] = size(gnd);
        Data.LabeledIndex = f(gnd, labelrate);
        Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
    end
    if strcmp(datafile, 'ORL.mat')
        load 'ORL.mat';
        fea = NormalizeFea(fea);
        Data.data = fea;
        Data.GroundTruth = gnd;
        [row, col] = size(gnd);
        Data.LabeledIndex = f(gnd, labelrate);
        Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
    end
    if strcmp(datafile, 'Yale.mat')
        load 'Yale.mat';
        fea = NormalizeFea(fea);
        Data.data = fea;
        Data.GroundTruth = gnd;
        [row, col] = size(gnd);
        Data.LabeledIndex = f(gnd, labelrate);
        Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
    end
    if strcmp(datafile, 'wine.mat')
        load 'wine.mat';
%         fea = NormalizeFea(fea);
        Data.data = data;
        Data.GroundTruth = label;
        [row, col] = size(label);
        Data.LabeledIndex = f(label, labelrate);
        Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
    end
    if strcmp(datafile, 'breast.mat')
        load 'breast.mat';
%         fea = NormalizeFea(fea);
        Data.data = data;
        Data.GroundTruth = label;
        [row, col] = size(label);
        Data.LabeledIndex = f(label, labelrate);
        Data.UnlabeledIndex = setxor((1:row), Data.LabeledIndex);
    end
end

function  index = f(x, rate)
%rand('seed', 1001);
labelnum = max(x);   %最大类别数
%%%% 打散x中原来的顺序
xind = randperm(length(x));
x = x(xind);
count = [];

%globalind = 0;
for i = 1:labelnum
    %formal
    %     eachnum = numel(find(x == i));
    %     randnum = sort(randi([globalind + 1, globalind + eachnum],floor(rate * eachnum), 1));
    %     count = [count; randnum];
    %     globalind = globalind+eachnum;
    %  later
    numindex = find(x == i);
    eachnum=floor(length(numindex) * rate);
    randnum = numindex(1:eachnum);
    count = [count; randnum];
end
index = count;
end