#%%
from datetime import datetime
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
import numpy as np
from difflib import get_close_matches
#%%
def get_date_string():
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


#%%
#data_file = "C:/Users/Jess/Dropbox/UTokyo/Research/Cellulose/code/visualization/input_data/20230824_cnf_flowcurve.xlsx"
#data_file = "./input_data/20230911_cnf_flowcurve_v1.xlsx"

samples =["P3-1 1", "P3-2 1", "P3-3 1", "P20-1 2", "P20-2 3", "P20-4 1", "P40-1 1", "P40-2 2", "P40-3 1", "1-1 2", "1-2 1", "1-3 1", "2-1 1", "2-3 1", "2-4 1", "3-1 1", "3-2 2", "3-3 1"]
names = ["Meas. Pts.", "Shear Rate",	"Shear Stress",	"Viscosity",	"Speed",	"Torque",	"Status"]
write_output = False

data_path = "./split_data"
data = pd.DataFrame()


for data_file in os.listdir(data_path):
    try:
        sample_name = data_file.split('.')[0]
        if sample_name in samples:
            print(sample_name)
            sample_data = pd.read_csv(os.path.join(data_path, data_file), header = 14, sep = ';')
            sample_data.columns = names
            sample_data["Full Sample"] = sample_name
            sample_data["Sample Name"] = sample_name.split("-")[0]     
            print(sample_data.head())
            data = pd.concat([data,sample_data], ignore_index = True)
    except:
        pass
    

#%%

#scatter = ax.plot(sample_data["Shear Rate"], sample_data["Viscosity"], label = sample_name)
fig, ax = plt.subplots()
print(data)

samples = ["P3", "1", "2", "P20", "3", "P40"] 
long_samples = ["P3", "P3 + P20", "P3 + P40", "P20", "P20 + P40", "P40"]

fit_data_path = "/Users/jessp/Dropbox/UTokyo/Research/Cellulose/code/rheology/fitting_data"

for i, sample_name in enumerate(samples):
    subset = data.query('`Sample Name` == @sample_name')
    print("subset:", subset)
    x = []
    y=[]
    z=[]
    err=[]
    
    for shear_rate in sorted(subset["Shear Rate"].unique()):
        x.append(shear_rate)
        
        mean = subset.loc[subset["Shear Rate"] == shear_rate,'Viscosity'].mean(axis=0)
        y.append(mean)
        ste =subset.loc[subset["Shear Rate"] == shear_rate,'Viscosity'].sem(axis=0)
        err.append(ste)
        
    errorbar= ax.errorbar(x, y, yerr=err,fmt='none' ,ecolor = 'black', capsize = 3)
    print(type(sample_name))
    scatter = ax.plot(x,y, label = long_samples[i])
    csv_file_path = fit_data_path + "/" + long_samples[i] + ".csv"
    # Open the CSV file for writing
    if write_output:
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write a header row if needed
            # writer.writerow(['x', 'y'])  # Uncomment this line to add headers
            # Write data rows
            for i in range(len(x)):
                writer.writerow([x[i], y[i], err[i]])
        
    

plt.legend(fontsize=8)
ax.set_xlabel('Shear rate [1/s]')
ax.set_ylabel('Viscosity [Pa/S]')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim([0.1,1000])
ax.set_title('Commercial Rheometer Measurements')
plt.legend()
plt.show()



#%%

fig.savefig("./output_plots/" + get_date_string() + "_viscosities.png", dpi=500)

#plt.savefig("./output_plots/viscosities.png")


# %%
