# import os
# import yaml

# INV = True

# def create_yaml_from_directory(directory_path, output_file):
#     # Get all file names in the directory
#     file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

#     if INV:
#         inv_files = [f for f in os.listdir(directory_path+'/inv') if os.path.isfile(os.path.join(directory_path+'/inv', f))]
      

#     # Calculate equal weight for each file
#     num_files = len(file_names)
#     if INV:
#         num_files += len(inv_files)
#     if num_files == 0:
#         raise ValueError("No files found in the directory")
    
#     weight = 1 / num_files

#     # Construct the data structure for the YAML file
#     yaml_data = {"motions": 
#                  [{"file": file_name, "weight": weight} for file_name in file_names]
#                  +
#                  [{"file": './inv/'+file_name, "weight": weight} for file_name in file_names]
#                  }

#     # Write to YAML file
#     with open(output_file, 'w+') as file:
#         yaml.dump(yaml_data, file, default_flow_style=False)

# # Usage
# directory_path = './a1_recording_processed'  # Replace with your directory path
# output_file = '.all_inv.yaml'  # Replace with your desired output file name
# create_yaml_from_directory(directory_path, output_file)


import os
import yaml

INV = True

def create_yaml_from_directories(directory_paths, output_file):
    yaml_data = {"motions": []}
    total_num_files = 0

    for directory_path in directory_paths:
        # Get all .npy file names in the directory with the full path appended
        file_names = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.npy')]
        if INV:
            # Also include inv .npy files with './inv/' directory and full path
            inv_directory = os.path.join(directory_path, 'inv')
            if os.path.exists(inv_directory):  # Check if the inv directory exists
                inv_files = [os.path.join(inv_directory, f) for f in os.listdir(inv_directory) if os.path.isfile(os.path.join(inv_directory, f)) and f.endswith('.npy')]
                file_names += inv_files

        total_num_files += len(file_names)
        # Append file data to yaml_data with full paths
        yaml_data["motions"].extend([{"file": file_name, "weight": 0} for file_name in file_names])  # Temporarily set weight to 0


    if total_num_files == 0:
        raise ValueError("No files found in the directories")

    weight = 1 / total_num_files
    # Update the weight for each file
    for motion in yaml_data["motions"]:
        motion["weight"] = weight

    # Write to YAML file
    with open(output_file, 'w+') as file:
        yaml.dump(yaml_data, file, default_flow_style=False)

# Usage
directory_paths = ['./a1_recording_processed', './a1_complex_processed', './dog_mocap_processed']  # Replace with your directory paths
output_file = 'all_inv.yaml'  # Replace with your desired output file name
create_yaml_from_directories(directory_paths, output_file)
