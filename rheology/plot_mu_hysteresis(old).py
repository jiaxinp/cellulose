import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the folder containing CSV files
folder_path = 'rheology/split_data/20240226'

# List to store data frames from each CSV file
data_frames = []
samples = []
groups = {}
headers = ["Meas. Pts.",	"Shear Rate",	"Shear Stress",	"Viscosity",	"Speed",	"Torque",	"Status"]

# Iterate over each file in the folder
for file_name in sorted(os.listdir(folder_path)):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file into a DataFrame and append it to the list
        try:
            df = pd.read_csv(file_path,skiprows=18,delimiter=';',header=None, names = headers, skipfooter=1)
            sample =file_name.split("_",1)[1].split('.')[0]
            print(sample)
            group = file_name.split("_")[1]
            print(group)
            if group not in groups:
                groups[group] = []
            else:
                groups[group].append(df)
            samples.append(sample)
        except:
            pass

# Plot the data from each DataFrame

print(groups)
print(samples)

group_df = pd.concat(groups)
mean_viscosity_over_time = group_df.groupby(0)["Viscosity"].mean()
plt.plot(mean_viscosity_over_time.index, mean_viscosity_over_time.values)

for i,df in enumerate(data_frames):
    df_filtered = df[df['Viscosity'] > 0]
    plt.plot(df_filtered["Shear Rate"], df_filtered['Viscosity'], label=samples[i])

# Add labels and legend
plt.xscale('log')
plt.xlabel("Shear Rate")
plt.xlim(left=0.01) 
plt.yscale('log')
plt.ylabel('Viscosity')
plt.title('Viscosity against shear rate for different length CNF')
for group, samples in groups.items():
    lines = [plt.Line2D([0], [0], color='black', linestyle='-') for _ in range(len(samples))]
    plt.legend(lines, samples, title=group, loc='right', bbox_to_anchor=(1, 1))

# Show or save the graph
plt.show()
# plt.savefig('plot.png')  # Uncomment this line to save the plot as an image
