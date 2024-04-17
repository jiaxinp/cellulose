
from utils.get_date_string import get_date_string
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from difflib import get_close_matches

#data_file = "C:/Users/Jess/Dropbox/UTokyo/Research/Cellulose/code/visualization/input_data/20230824_cnf_flowcurve.xlsx"
data_file = "./input_data/20230824_cnf_flowcurve_v1.xlsx"
data = pd.read_excel(data_file)

print(data.head())
samples = data.Sample.unique()
print(samples)
sample_names =["P40-1","P40-2","P40-3","P20-1","P20-2","P20-3","P3-1","P3-2","P3-3" ]
sample_names.reverse()
fig, ax = plt.subplots()


for i, sample in enumerate(np.flip(samples)):
    
    plt.plot(data.loc[data['Sample'] == sample, 'Shear Rate'], data.loc[data['Sample'] == sample, 'Viscosity'], '.-', label = sample_names[i])


plt.legend(fontsize=8)
ax.set_xlabel('Shear rate 1/s')
ax.set_ylabel('Viscosity Pa/S')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_title('Commercial Rheometer Measurements')
plt.savefig("./output_plots/" + get_date_string() + "_viscosities.png")
#plt.savefig("./output_plots/viscosities.png")
plt.show()