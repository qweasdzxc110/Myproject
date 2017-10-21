clear;
load 'TDT2.mat'

labelnum = max(gnd);   %最大类别数
count = [];
rand('seed', 1001);
globalind = 1;
for i = 1: labelnum
eachnum = numel(find(gnd == i));
randnum = sort(randi([globalind, globalind + eachnum],floor(0.1 * eachnum), 1));
count = [count; randnum];
globalind = globalind+eachnum;
end
count
 