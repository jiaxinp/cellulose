clear
clc
close all

data = readmatrix("input_data/20230911_cnf_flowcurve_v1.xlsx")
skip =5

for i = 0:9
    start = i*25+1+skip
    x = data(start: start + 24-skip, 3);
    y = data(start: start +24-skip, 5)*1000;
    fit_carreau(x,y)
end
