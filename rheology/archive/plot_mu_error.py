
from datetime import datetime
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from difflib import get_close_matches

def get_date_string():
    now = datetime.now()
    return now.strftime("%m%d%Y-%H%M%S")



#data_file = "C:/Users/Jess/Dropbox/UTokyo/Research/Cellulose/code/visualization/input_data/20230824_cnf_flowcurve.xlsx"
data_file = "./input_data/20230911_cnf_flowcurve_v1.xlsx"
data = pd.read_excel(data_file)


data["sample_names"]= data['Sample'].str.split('_').str[0]
data["sample_no"]= data['Sample'].str.split('_').str[1]
print(data.head())




fig, ax = plt.subplots()

names_dict= {
    "1": "P3",
    "2":"P20",
    "3":"P40"
    }

def get_sample_name(name):
    if type(name) == float:
        return names_dict[str(name)]
    else:
        return names_dict[name.split("-")[0]]

for sample_name in data['sample_names'].unique():
    subset = data.query('`sample_names` == @sample_name')
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
    scatter = ax.plot(x,y, label = get_sample_name(sample_name))

plt.legend(fontsize=8)
ax.set_xlabel('Shear rate 1/s')
ax.set_ylabel('Viscosity Pa/S')
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlim([0.1,100])
ax.set_title('Commercial Rheometer Measurements')
plt.savefig("./output_plots/" + get_date_string() + "_viscosities.png")
#plt.savefig("./output_plots/viscosities.png")
plt.show()