import pandas as pd
import numpy as np

# CLS_PATH = '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/utils/class_labels.pkl'
# df = pd.read_pickle(CLS_PATH)
# df['File_Name'] = df['File_Name'].replace('./data', '', regex=False)
# class_labels = df['Class_Labels'].values

CLS_PATH = '/home/mcarroll/Documents/cdt-1/ASE-Atlas/ase/utils/class_labels_all_motions.pkl'
df = pd.read_pickle(CLS_PATH)
df = df.sort_values(by='Cluster_Index', ascending=True)
first_rows = df.groupby('Cluster_Index').first()

print(first_rows)
input()
class_labels = first_rows['Class_Labels'].values




# Converting tuples back to np arrays if necessary
unique_arrays = [np.array(t) for t in class_labels]

# Stack the unique arrays into a single np array
final_array = np.vstack(unique_arrays)

# Save the array to a binary file (.npy)
np.save('./unique_classes_all_motions.npy', final_array)

print(final_array)
