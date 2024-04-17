D = 1 %diameter of channel in mm
Q_cnf = 4.18 %flow rate of cnf in ml/h
Q_h2o = 4.97 % flow rate of h2o in ml/h
Q_hcl = 29.88 %flow rate of acid in ml/j

Q_total = (Q_cnf + Q_h2o *2 + Q_hcl *2) /(1e6 * 3600) %total flow rate in m3/s

A = ((D*1e-3)/2)^2 * pi

v = Q_total/A