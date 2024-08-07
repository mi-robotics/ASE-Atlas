
# import os
# import yaml

# INV = True

# def create_yaml_from_directories(directory_paths, output_file):
#     yaml_data = {"motions": []}
#     total_num_files = 0

#     for directory_path in directory_paths:
#         # Get all .npy file names in the directory with the full path appended
#         file_names = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f)) and f.endswith('.npy') and not 'biped' in f and  not'handstand' in f and  not'jump' in f and  not'step' in f and  not'bound' in f] 
#         if INV:
#             # Also include inv .npy files with './inv/' directory and full path
#             inv_directory = os.path.join(directory_path, 'inv')
#             if os.path.exists(inv_directory):  # Check if the inv directory exists
#                 inv_files = [os.path.join(inv_directory, f) for f in os.listdir(inv_directory) if os.path.isfile(os.path.join(inv_directory, f)) and f.endswith('.npy') and not 'biped' in f and  not'handstand' in f and  not'jump' in f and  not 'step' in f and  not'bound' in f]
#                 file_names += inv_files

#         total_num_files += len(file_names)
#         # Append file data to yaml_data with full paths
#         yaml_data["motions"].extend([{"file": file_name, "weight": 0} for file_name in file_names])  # Temporarily set weight to 0


#     if total_num_files == 0:
#         raise ValueError("No files found in the directories")

#     weight = 1 / total_num_files
#     # Update the weight for each file
#     for motion in yaml_data["motions"]:
#         motion["weight"] = weight

#     # Write to YAML file
#     with open(output_file, 'w+') as file:
#         yaml.dump(yaml_data, file, default_flow_style=False)

# # Usage
# directory_paths = ['./a1_recording_processed', './a1_complex_processed', './dog_mocap_processed']  # Replace with your directory paths
# output_file = 'all_no_jump_inv.yaml'  # Replace with your desired output file name
# create_yaml_from_directories(directory_paths, output_file)

import os
import yaml

INV = True

def read_exclusion_file(exclusion_file_path):
    """Read file paths to exclude from a file."""
    with open(exclusion_file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def create_yaml_from_directories(directory_paths, output_file, exclusion_file_path):
    yaml_data = {"motions": []}
    total_num_files = 0

    # Read the exclusion list
    excluded_file_paths = read_exclusion_file(exclusion_file_path)

    for directory_path in directory_paths:

        # Get all .npy file names in the directory with the full path appended
        file_names = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                      if os.path.isfile(os.path.join(directory_path, f)) 
                      and f.endswith('.npy') 
                    #   and not 'rand' in f
                    #   and not 'dog' in f
                    #   and not 'biped' in f 
                    #   and not 'handstand' in f 
                    #   and not 'jump' in f 
                    #   and not 'step' in f 
                    #   and not 'bound' in f 
                      and os.path.join(directory_path, f) not in excluded_file_paths] 
        
  
        if INV:
            # Also include inv .npy files with './inv/' directory and full path
            inv_directory = os.path.join(directory_path, 'inv')
            if os.path.exists(inv_directory):  # Check if the inv directory exists
                inv_files = [os.path.join(inv_directory, f) for f in os.listdir(inv_directory) 
                             if os.path.isfile(os.path.join(inv_directory, f)) 
                             and f.endswith('.npy') 
                            #  and not 'rand' in f
                            #  and not 'dog' in f
                            #  and not 'biped' in f 
                            #  and not 'handstand' in f 
                            #  and not 'jump' in f 
                            #  and not 'step' in f 
                            #  and not 'bound' in f
                             and os.path.join(inv_directory, f) not in excluded_file_paths]
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

# Usage example
exclusion_file_path = './bad_motions.txt'  # This should be the path to your file containing the filepaths to exclude
# directory_paths = ['./train_data_seq_50']  # Replace with your directory paths
directory_paths = [
    # './a1_v3_processed',
    # './a1_recording_processed',
    './dog_mocap_processed',
    # './continual/back',   
    # './continual/left',
    # './continual/right',
    # # './continual/stand',
    # './continual/backflip',    
    # './continual/frontflip',
    ]  # Replace with your directory paths

output_file = './all_play.yaml'  # Replace with your desired output file name
create_yaml_from_directories(directory_paths, output_file, exclusion_file_path)

