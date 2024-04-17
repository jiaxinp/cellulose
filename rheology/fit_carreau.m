function [coeffvals, gof] = fit_carreau(x,y,name)

% Define Start points, fit-function and fit curve
x0 = [5 1756 16.16 0.56]; 
lb=[1 50 2 0];
ub = [ 10 3000 300 2];
fitfun = fittype( @(nu_inf,nu_0,tau,n,x) (nu_inf + (nu_0 - nu_inf)*((1+(tau*x).^2).^(n-1/2) )));
[fitted_curve,gof] = fit(x,y,fitfun,'StartPoint',x0, 'Lower',lb, 'Upper', ub)
% Save the coeffiecient values for a,b,c and d in a vector
coeffvals = coeffvalues(fitted_curve);
% Plot results
figure()
loglog( x, y, 'r+')
hold on
plot(x,fitted_curve(x))
formatSpec = '%.2f';
dim = [0.2 0.2 0.3 .3];
str = [name, "\eta_\infty: " + num2str(coeffvals(1),formatSpec), "\eta_0: " + num2str(coeffvals(2),formatSpec), "\tau: "+ num2str(coeffvals(3),formatSpec), "n: "+ num2str(coeffvals(4),formatSpec), "rsquare: " + num2str(gof.rsquare)];
annotation('textbox',dim,'String',str,'FitBoxToText','on','Interpreter',"tex");
xlabel("Shear rate 1/s")
ylabel('Viscosity Pa/S')
title(['Fitting ', name])
file_name = "./output_plots/20230919_" + name +".jpg"
saveas( gcf,file_name )
hold off

end