dz = (100)*1e-3; %distance between sensors in mm
R = (0.5)*1e-3/2; %radius in mm converted to meters
Q = 1  * 1e-6/ 60 ;
mu = 1e-2

del_P = mu/1000 * (8*Q*dz) /(pi*R^4) %Pa

disp(del_P / (0.01))
