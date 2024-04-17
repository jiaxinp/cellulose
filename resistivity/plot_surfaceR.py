import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)
output_path = "output/"

plt.rcParams['font.family'] = 'Segoe UI'
title_font = {'fontname':'Garamond', 'fontsize':14, 'fontweight':'bold'}

#plot reference data
ref_file_path = 'input/hiresta.xlsx'

# Load the Excel file into a DataFrame
df_ref = pd.read_excel(ref_file_path)
print(df_ref)

# Show or save the graph
plt.show()

#plot measurement data
file_path = 'input/20240312_k2636b.xlsx'
file_name = file_path.split("/")[1].split(".")[0]

# Load the Excel file into a DataFrame
df = pd.read_excel(file_path)
print(df)

samples = df["Sample"].unique()
averages =[]
ref_averages = []
errors=[]
ref_errors =[]

for sample in samples:
    # Slice the DataFrame to include only rows corresponding to the current sample
    sample_df = df[df["Sample"] == sample]
    
    # Calculate the average value for the "SR" column of the current sample
    sample_ave = sample_df["SR"].mean()
    sample_err = sample_df["SR"].std()/math.sqrt(sample_df.shape[0])
    averages.append(sample_ave)
    errors.append(sample_err)
    ref =df_ref[df_ref["Sample"] == sample]
    ref_average = ref["average"].iloc[0]
    ref_averages.append(ref_average)
    ref_error = ref["stdev"].iloc[0]/math.sqrt(3)
    ref_errors.append(ref_error)
    
plt.bar(samples, averages, yerr=errors,capsize=5)
plt.errorbar(samples, ref_averages, yerr=ref_errors, fmt='o', color='red', label='Reference Points') 

plt.xlabel("Sample")
plt.yscale('log')
plt.ylabel(r'Surface Resistivity [Pa$\cdot$ s]')
plt.title('Surface Resistivity Measurement of CNF Films', **title_font)
#plt.legend(["P3 = 636nm", "P20 = 414nm","P40 = 318nm" ])
plt.tight_layout()
plt.savefig(output_path+file_name + ".png",dpi=500)
plt.show()


