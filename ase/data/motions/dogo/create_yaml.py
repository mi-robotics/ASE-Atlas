import os
import yaml

def create_yaml_from_directory(directory_path, output_file):
    # Get all file names in the directory
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

    # Calculate equal weight for each file
    num_files = len(file_names)
    if num_files == 0:
        raise ValueError("No files found in the directory")
    weight = 1 / num_files

    # Construct the data structure for the YAML file
    yaml_data = {"motions": [{"file": file_name, "weight": weight} for file_name in file_names]}

    # Write to YAML file
    with open(output_file, 'w') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

# Usage
directory_path = '../dog_mocap_processed'  # Replace with your directory path
output_file = 'output.yaml'  # Replace with your desired output file name
create_yaml_from_directory(directory_path, output_file)