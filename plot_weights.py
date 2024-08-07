import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def plot_logs(jsonl_file_path):
    # Read the JSONL file and convert to a DataFrame
    data = []
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    
    # Convert list to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure data types are correct, especially if weights or probs are in different formats
    df['weights'] = pd.to_numeric(df['weights'], errors='coerce')
    df['probs'] = pd.to_numeric(df['probs'], errors='coerce')  # Adjust if probs are more complex
    
    # Plotting
    classes = df['class'].unique()
    plt.figure(figsize=(12, 6))
    
    # Plot weights for each class
    for cl in classes:
        class_data = df[df['class'] == cl]
        plt.plot(class_data['iteration'], class_data['weights'], label=f'Class {cl} Weights')
    
    plt.title('Weights vs Iterations for Each Class')
    plt.xlabel('Iteration')
    plt.ylabel('Weights')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot probabilities if necessary (separate plot or combine as needed)
    plt.figure(figsize=(12, 6))
    for cl in classes:
        class_data = df[df['class'] == cl]
        plt.plot(class_data['iteration'], class_data['probs'], label=f'Class {cl} Probs', linestyle='--')
    
    plt.title('Probabilities vs Iterations for Each Class')
    plt.xlabel('Iteration')
    plt.ylabel('Probabilities')
    plt.legend()
    plt.grid(True)
    plt.show()



    # Plotting setup
    classes = df['class'].unique()
    color_map = get_cmap('viridis')  # Using a colormap that provides good visibility

    # Plot weights and probabilities for each class
    plt.figure(figsize=(16, 8))
    num_classes = len(classes)
    bar_width = 0.35  # Width of the bars

    # Determine the last iteration number
    last_iteration = df['iteration'].max()
    
    # Filter data for the last iteration
    last_data = df[df['iteration'] == last_iteration]

    # Collect weights and probabilities for the last iteration for each class
    weights = [last_data[last_data['class'] == cl]['weights'].values[0] for cl in classes]
    probs = [last_data[last_data['class'] == cl]['probs'].values[0] for cl in classes]

    # Create bar charts
    index = range(num_classes)  # Class indices as x
    fig, ax = plt.subplots()
    bars1 = ax.bar(index, weights, bar_width, label='Weights', color='b')
    bars2 = ax.bar([p + bar_width for p in index], probs, bar_width, label='Probabilities', color='g')

    ax.set_xlabel('Class')
    ax.set_ylabel('Values')
    ax.set_title('Weights and Probabilities by Class at Last Iteration')
    ax.set_xticks([p + bar_width / 2 for p in index])
    ax.set_xticklabels([f'Class {cl}' for cl in classes])
    ax.legend()

    plt.grid(True)
    plt.show()

# Use the function
plot_logs('./continual_logs.jsonl')
