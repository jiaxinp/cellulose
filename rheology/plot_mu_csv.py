import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
from utils.colors import colors as all_colors
from utils.custom_sort import custom_sort

# Path to the folder containing CSV files
folder_path = 'rheology/split_data/20240226'

data_name = folder_path.split("/")[-1]
output_path = 'rheology/output_plots'+ "/"+ data_name +"/"
plt.rcParams['font.family'] = 'Segoe UI'
title_font = {'fontname':'Garamond', 'fontsize':14, 'fontweight':'bold'}
#colors = ['black', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'yellow']
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# List to store data frames from each CSV file
data_frames = []

headers = ["Meas. Pts.",	"Shear Rate",	"Shear Stress",	"Viscosity",	"Speed",	"Torque",	"Status"]
skip_list = ["P20_1 1","P3_1 1"]
x_limits= {"P3": 1e-3, "P20": 1e-2, "P40": 1e-1 }
y_limits ={"P3": [0.013, 18], "P20": [0.0065,0.1], "P40": [5e-3, 0.028] } 
lengths ={"P3": 636, "P20": 414, "P40": 318 } 

# Iterate over each file in the folder
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file into a DataFrame and append it to the list
        try:
            df = pd.read_csv(file_path,skiprows=18,delimiter=';',header=None, names = headers, skipfooter=1)
            sample =file_name.split("_",1)[1].split('.')[0]
            group = file_name.split("_")[1]
            df['Sample'] = sample
            df['Group'] = group
        # Append the DataFrame to the list
            data_frames.append(df)
        except:
            pass

# Plot the data from each DataFrame

combined_df = pd.concat(data_frames, ignore_index=True)

group_names = sorted(combined_df['Group'].unique(),key=custom_sort)

for i,group in enumerate(group_names):
    group_df = combined_df[combined_df['Group'] == group]
    
    sample_names = group_df['Sample'].unique() 
    plt.figure()
    counter = 0
    
    for sample in sample_names:
        if sample in skip_list:
            continue
        
        sample_df = group_df[group_df['Sample'] == sample]
        df_filtered = sample_df[sample_df['Viscosity'] > 0]
        if counter%2:
            direction = "down"
            mark = "v-"
        else:
           direction = "up" 
           mark= "^-"
        
        new_label = group +" "+ direction +" "+ str(counter+1)
        plt.figure(2+i)
        plt.plot(df_filtered["Shear Rate"], df_filtered['Viscosity'],mark, label= new_label,color=colors[counter//2])
        plt.figure(1)
        if counter:
            new_label = None
        else:
            new_label = group + " = " + str(lengths[group]) + "nm"
            
        plt.plot(df_filtered["Shear Rate"], df_filtered['Viscosity'],mark, label= new_label, color= all_colors[i][1][counter//2])

        counter = counter +1
    plt.figure(2+i)
    plt.xscale('log')
    plt.xlabel("Shear Rate [1/s]")
    plt.xlim(left=x_limits[group])
    plt.ylim(y_limits[group])  
    plt.yscale('log')
    plt.ylabel(r'Viscosity [Pa$\cdot$ s]')
    plt.title('Viscosity Hysteresis for ' + group, **title_font) 
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{data_name}_{group}.png'), dpi =500)

# Add labels and legend
plt.figure(1)
plt.xscale('log')
plt.xlabel("Shear Rate [1/s]")
plt.xlim(left=0.01) 
plt.yscale('log')
plt.ylabel(r'Viscosity [Pa$\cdot$ s]')
plt.title('Viscosity against Shear Rate for Different CNF lengths', **title_font)
#plt.legend(["P3 = 636nm", "P20 = 414nm","P40 = 318nm" ])
plt.legend()
plt.tight_layout()
plt.savefig(output_path+ "all" + ".png",dpi=500)

# Show or save the graph
plt.show()
# plt.savefig('plot.png')  # Uncomment this line to save the plot as an image

# Plot hysteresis
plt.figure()