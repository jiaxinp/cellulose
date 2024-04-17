clear
clc
close all

path = "./fitting_data";
files = dir(strcat(path, '/*.csv'));
skip = 8

for i = 1:length(files)
    data = csvread(strcat(path,'/', files(i).name));
    x = data(skip: end, 1);
    y = data(skip: end, 2)*1000;
    fit_carreau(x,y, files(i).name(1:end-4))
end

