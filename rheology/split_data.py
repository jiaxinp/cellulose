import csv
import os

# Function to split a CSV file based on a delimiter line
def split_csv_by_delimiter(input_file, output_directory, delimiter_line):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    with open(input_file, mode='r', newline='',encoding='iso-8859-1') as csvfile:
        reader = csv.reader(csvfile)
        
        delimiter_found = False
        next_line_is_filename = False
        current_output_file = None
        
        for row in reader:
            if delimiter_line in row:
                delimiter_found = True
                next_line_is_filename = True
                if current_output_file:
                    current_output_file.close()
            elif next_line_is_filename:
                next_line_is_filename = False
                output_file_name = row[0].split(';')[-1] + ".csv"
                output_file = os.path.join(output_directory, output_file_name)
                current_output_file = open(output_file, mode='w', newline='')
                csv_writer = csv.writer(current_output_file)
                csv_writer.writerow(row)
            else:
                if delimiter_found and current_output_file:
                    csv_writer.writerow(row)
    
    
    if current_output_file:
        current_output_file.close()

if __name__ == "__main__":
    input_csv = "/Users/jessp/Dropbox/UTokyo/Research/Cellulose/Data/Rheometer/20240226_cnf_FlowCurve Data_20240226230535.csv"  # Replace with your input CSV file path
    output_dir = "/Users/jessp/Dropbox/UTokyo/Research/Cellulose/code/rheology/split_data/20240226"    # Replace with the directory where you want to save the split CSV files
    delimiter_line = "Data Series Information"  # Replace with the specific delimiter line words
    
    split_csv_by_delimiter(input_csv, output_dir, delimiter_line)
