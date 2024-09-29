import os
import numpy as np

# Function to load and preprocess all drillbit data
def load_data(data_dir):
    """
    Load all the data files for each drill bit and return them as a list of numpy arrays.
    Each array corresponds to the data for one drill bit (each file).
    """
    drillbit_data = []
    
    # Loop over all files in the data directory
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".TXT"):
            filepath = os.path.join(data_dir, filename)
            data = np.loadtxt(filepath)  # Load each file as a numpy array
            drillbit_data.append(data)
    print(f"Loaded {len(drillbit_data)} drill bit files.")
    # print( f"Data shape: {drillbit_data[0]}")
    return drillbit_data

# Function to normalize a sequence using min-max normalization
def normalize_sequence(sequence):
    """
    Normalize the sequence data (thrust force, torque) using min-max normalization.
    Each column (thrust force, torque) will be normalized between 0 and 1.
    """
    min_vals = np.min(sequence, axis=0)
    max_vals = np.max(sequence, axis=0)
    
    # Avoid division by zero if the max and min are the same
    normalized_sequence = (sequence - min_vals) / (max_vals - min_vals + 1e-8)
    
    # print(f"Normalized sequence from {min_vals} to {max_vals}")
    # print(f"Sequence shape: {sequence.shape}, Normalized sequence shape: {normalized_sequence.shape}")
    return normalized_sequence

# Function to generate temporal sequences (holes) from the drill bit data
def generate_sequences(drillbit_data):
    """
    Takes the list of drill bit data and generates temporal sequences (holes) 
    from each drill bit's data. The sequences are split by zeros (hole separators),
    and each sequence is normalized.
    """
    temporal_sequences = []
    
    for data in drillbit_data:
        start_idx = None
        for i, row in enumerate(data):
            # Detect sequences by looking at zeros (hole separators)
            if row[0] == 0 and row[1] == 0:
                if start_idx is not None:  # End of a sequence
                    sequence = data[start_idx:i]  # Extract the sequence
                    if len(sequence) > 0:
                        # Normalize the sequence
                        normalized_sequence = normalize_sequence(sequence)
                        temporal_sequences.append(normalized_sequence)  # Store the normalized sequence
                    start_idx = None
            else:
                if start_idx is None:
                    start_idx = i  # Start of a new sequence
    print(f"Generated {len(temporal_sequences)} normalized temporal sequences (holes).")
    
    # for(i, seq) in enumerate(temporal_sequences):
    #     print(f"Sequence {i} shape: {seq.shape}")
    return temporal_sequences

if __name__ == "__main__":
    data_dir = '../data'  # Path to the data folder
    drillbit_data = load_data(data_dir)
    temporal_sequences = generate_sequences(drillbit_data)
    
    
    print(f"Loaded {len(drillbit_data)} drill bit files.")
    print(f"Generated {len(temporal_sequences)} normalized temporal sequences (holes).")
    print(f"Sequence 0 shape: {temporal_sequences[0].shape}")
    print(f"Sequence 1 shape: {temporal_sequences[1].shape}")   
