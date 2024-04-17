% Some data - replace it with yours (its from an earlier project)
x = [
0.0681
0.1
0.147
0.215
0.316
0.464
0.681
1
1.47
2.15
3.16
4.64
6.81
10
14.7
21.5
31.6
46.4
68.1
100];
y = [
0.2240
0.2160
0.2100
0.1900
0.1700
0.1500
0.1330
0.1170
0.1030
0.0897
0.0774
0.0663
0.0569
0.0486
0.0415
0.0353
0.0300
0.0255
0.0217
0.0186
] *1000;
% Define Start points, fit-function and fit curve
x0 = [5 1756 16.16 0.56]; 
lb=[1 100 0 0];
ub = [ 10 5000 200 1]
fitfun = fittype( @(nu_inf,nu_0,tau,n,x) (nu_inf + (nu_0 - nu_inf)*((1+(tau*x).^2).^((n-1)/2)) ));
[fitted_curve,gof] = fit(x,y,fitfun,'StartPoint',x0, 'Lower',lb, 'Upper', ub)
% Save the coeffiecient values for a,b,c and d in a vector
coeffvals = coeffvalues(fitted_curve);
% Plot results


loglog( x, y, 'r+')

hold on
plot(x,fitted_curve(x))
hold off