from preprocessing import load_data, generate_sequences
from hmm_training import initialize_hmms, competitive_learning
from utils import save_hmms

if __name__ == "__main__":
    data_dir = '../data'  # Data directory for drill bit files

    # Step 1: Load and preprocess the data
    drillbit_data = load_data(data_dir)
    temporal_sequences = generate_sequences(drillbit_data)
    print(f"Generated {len(temporal_sequences)} temporal sequences.")
    
    # Step 2: Initialize and train 10 HMMs
    hmms = initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2)
    
     # Perform competitive learning, ensuring sequence length is at least 4
    hmms, top_hmms, hmm_win_count = competitive_learning(temporal_sequences, hmms, max_epochs=20, min_sequence_length=4, tolerance=1e-4, max_refits=5)

#     # After competitive learning is complete
#     hmms, hmm_win_counts, hmm_sequence_dict = competitive_learning(
#     temporal_sequences, hmms, max_epochs=10, min_sequence_length=4, tolerance=1e-4, max_refits=5)

# # Print the sequences won by each HMM and the win counts
#     for hmm_idx, seq_dict in hmm_sequence_dict.items():
#         print(f"HMM {hmm_idx} won the following sequences:")
#         for seq_idx, win_count in seq_dict.items():
#             print(f"  Sequence {seq_idx} was won {win_count} times")

    
    
    # Step 4: Save trained HMMs
    save_hmms(hmms, output_dir='./results/models/')

    # Step 5: Evaluate and select the top 3 HMMs
    save_hmms(top_hmms, output_dir='./results/top_hmms/')
    print("HMM training and selection completed.")
