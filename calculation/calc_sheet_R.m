t = 30e-6; % thickness
rho_max = 1e14; %maximum sheet resistance
I_min = 1e-9; % minimum current
R_max = (rho_max)/(pi/log(2)); %Maximum measured resistance

V_max = R_max*I_min % Maximum voltage difference

Rs = []
samples = []
sample= "Na SMU";
I=13e-9;
V =17;

rho= pi/log(2) * V/I
Rs = [Rs rho];
samples = [samples sample];
sample = "Na8888 SMU";
I = 0.2e-9
V= 18

rho= pi/log(2) * V/I
Rs = [Rs rho];
samples = [samples sample];

sample = "P8888 SMU";
disp("P8888 SMU")
I = 0.3e-9
V= 17.5

rho= pi/log(2) * V/I
Rs = [Rs rho];
samples = [samples sample];

sample = "P8888 GS610";
disp("P8888 GS610")
I = 0.2e-9
V= 110

rho= pi/log(2) * V/I
Rs = [Rs rho];
samples = [samples sample];

sample = "N8888 GS610";
disp("N8888")
I = 0.1e-9
V= 110

rho= pi/log(2) * V/I
Rs = [Rs rho];
samples = [samples sample];

sample = "Na K2636B";
disp("Na")
R = 50e6

rho= pi/log(2) * R
Rs = [Rs rho];
samples = [samples sample];

figure()

bar(samples, log10(Rs) )
ylim([6,14])
xlabel("Sample")
ylabel("Sheet Resistivity Order of Magnitude")
saveas(gcf,"20240214 sheetresistance.jpg")