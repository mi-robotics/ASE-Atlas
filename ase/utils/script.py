import pandas as pd
import numpy as np

CLS_PATH = '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/utils/class_labels.pkl'
df = pd.read_pickle(CLS_PATH)
df['File_Name'] = df['File_Name'].replace('./data', '', regex=False)
class_labels = df['Class_Labels'].values

CLS_PATH = '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/utils/class_labels.pkl'
df = pd.read_pickle(CLS_PATH)
df['File_Name'] = df['File_Name'].replace('./data', '', regex=False)
class_labels = df['Class_Labels'].values


# Using a set to store unique arrays as tuples
unique_tuples = set(tuple(array) for array in class_labels)

# Converting tuples back to np arrays if necessary
unique_arrays = [np.array(t) for t in unique_tuples]

# Stack the unique arrays into a single np array
final_array = np.vstack(unique_arrays)

# Save the array to a binary file (.npy)
np.save('./unique_classes.npy', final_array)

print(final_array)
