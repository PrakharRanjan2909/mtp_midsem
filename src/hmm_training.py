import random
import numpy as np
from hmmlearn import hmm
from kl_divergence_implementations import calculate_kl_divergence_matrix
from preprocessing import generate_sequences_each_file



# Function to initialize HMMs with random subsets
def initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2):
    """
    Initialize HMMs by training them with random subsets of the temporal sequences
    for a few iterations (e.g., 2 iterations).
    """
    hmms = []
    
    # Initialize HMMs
    for i in range(num_hmms):
        model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=num_iterations)
        
        # Randomly select a subset of sequences to train the HMM initially
        random_subset = random.sample(temporal_sequences, 5)
        lengths = [len(seq) for seq in random_subset]
        combined_data = np.vstack(random_subset)  # Combine all sequences into one for training
        
        # Train the HMM on this random subset for a couple of iterations
        model.fit(combined_data, lengths)
        hmms.append(model)
    
    return hmms

# Function to initialize HMMs with a fixed number of sequences per HMM
def initialize_hmms_limited_sequences(temporal_sequences_2d_list, num_hmms=10, num_sequences_per_hmm=10, num_iterations=2):
    """
    Initialize HMMs by assigning a fixed number of sequences (e.g., 10) to each HMM
    from the temporal sequences in an orderly fashion across files.

    Parameters:
        temporal_sequences_2d_list (list of list): 2D list of temporal sequences for each drill bit file.
        num_hmms (int): Number of HMMs to initialize.
        num_sequences_per_hmm (int): Number of sequences to assign to each HMM for pretraining.
        num_iterations (int): Number of iterations for initial training.
        
    Returns:
        list: List of trained HMMs.
    """
    hmms = []

    # Initialize empty lists to store sequences for each HMM
    sequences_for_hmms = [[] for _ in range(num_hmms)]
    lengths_for_hmms = [[] for _ in range(num_hmms)]

    # Assign a fixed number of sequences to each HMM
    seq_count = 0
    for file_sequences in temporal_sequences_2d_list:
        for sequence in file_sequences:
            # Assign to the current HMM in a round-robin fashion
            hmm_idx = seq_count % num_hmms
            if len(sequences_for_hmms[hmm_idx]) < num_sequences_per_hmm:
                sequences_for_hmms[hmm_idx].append(sequence)
                lengths_for_hmms[hmm_idx].append(len(sequence))
            
            seq_count += 1
            # Stop if all HMMs have received the required number of sequences
            if all(len(seqs) >= num_sequences_per_hmm for seqs in sequences_for_hmms):
                break
        if all(len(seqs) >= num_sequences_per_hmm for seqs in sequences_for_hmms):
            break
   
    for i in range(num_hmms):
        model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=num_iterations)

        # Combine sequences for this HMM into a single training dataset
        if sequences_for_hmms[i]:
            combined_data = np.vstack(sequences_for_hmms[i])
            lengths = lengths_for_hmms[i]
            
            # Train the HMM on the combined sequences
            model.fit(combined_data, lengths)
            hmms.append(model)
            print(f"HMM {i+1} initialized with {len(sequences_for_hmms[i])} sequences.")

    return hmms

# Function for competitive learning using HMMs with a stopping criterion
def competitive_learning(temporal_sequences, hmms, max_epochs=20, min_sequence_length=4, tolerance=1e-4, max_refits=5):
    """
    Perform competitive learning on the temporal sequences with the initialized HMMs.
    Each sequence is assigned to the HMM that maximizes the log-likelihood, and the
    winning HMM updates its parameters.
    
    Only sequences that are at least as long as the number of HMM states are used for training.
    Includes convergence checks based on log-likelihood improvements.
    Limits the number of refits for each sequence to avoid endless looping.
    """
    refit_counts = [0] * len(temporal_sequences)  # Track how many times each sequence is refit
    
    dict = {}
    hmm_win_count = {i: 0 for i in range(len(hmms))}  # Dictionary to store win counts for each HMM
    
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        
        # Track total log-likelihood change to check for convergence
        total_log_likelihood_change = 0
        #randomise the temporal sequences
        random.shuffle(temporal_sequences)
        # random.shuffle(temporal_sequences)
        
        for idx, seq in enumerate(temporal_sequences):
            


            seq = np.array(seq)  # Ensure the sequence is a numpy array

            
            # Skip sequences that are too short for the number of HMM states
            if len(seq) < min_sequence_length:
                print(f"Skipping sequence of length {len(seq)} (too short for HMM with {min_sequence_length} states)")
                continue
            
            # # Skip refitting if a sequence has been refit too many times
            # if refit_counts[idx] >= max_refits:
            #     print(f"Skipping sequence {idx} as it has been refit {max_refits} times")
            #     continue
            
            best_hmm = None
            best_log_likelihood = float('-inf')
            best_hmm_idx = -1
            
            # Evaluate each HMM on the sequence
            for hmm_idx, model in enumerate(hmms):
                log_likelihood = model.score(seq)  # Calculate log-likelihood for the sequence
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood = log_likelihood
                    best_hmm = model
                    best_hmm_idx = hmm_idx
            
            # Combine the current sequence into a 2D array (fit expects 2D)
            seq = seq.reshape(-1, 2)  # Reshape to (n_samples, n_features)
            
            # Save the previous parameters to check for convergence
            old_params = best_hmm.transmat_.copy()
            
            # Update the winning HMM with the sequence
            # Using EM algorithm to update the parameters

            best_hmm.fit(seq, [len(seq)])  # Fit using the sequence and its length
            refit_counts[idx] += 1  # Increment refit counter for the sequence
            
            print(f"HMM {best_hmm_idx} won the sequence {idx}")


            # Update the win count for the HMM
            if best_hmm_idx in hmm_win_count:
                hmm_win_count[best_hmm_idx] += 1
            else:
                hmm_win_count[best_hmm_idx] = 1
            
            for key in hmm_win_count:
                print(f"HMM {key} win count: {hmm_win_count[key]}")

            
            if best_hmm_idx in dict:
                dict[best_hmm_idx] += 1
            else:
                dict[best_hmm_idx] = 1
            
            for key in dict:
                print(f"Key: {key}, Value: {dict[key]}")
            
            # Compute log-likelihood change to track improvement
            new_log_likelihood = best_hmm.score(seq)
            likelihood_change = np.abs(new_log_likelihood - best_log_likelihood)
            total_log_likelihood_change += likelihood_change
            
            # Check if the HMM has converged based on parameter change or log-likelihood change
            if np.max(np.abs(best_hmm.transmat_ - old_params)) < tolerance:
                print(f"HMM {best_hmm_idx} has converged on sequence {idx}.")
        
        print(f"Total log-likelihood change in epoch {epoch + 1}: {total_log_likelihood_change}")
        
        # # Early stopping criterion based on total log-likelihood change across the epoch
        # if total_log_likelihood_change < tolerance:
        #     print(f"Convergence reached after {epoch + 1} epochs.")
        #     break

    # Sort HMMs based on the number of wins
    top_hmms_idx = sorted(hmm_win_count, key=hmm_win_count.get, reverse=True)[:3]
    top_hmms = [hmms[i] for i in top_hmms_idx]
    
    print(f"Top 3 HMMs based on win count: {top_hmms_idx}")
    
    return hmms, top_hmms, hmm_win_count

def select_top_hmms_with_kl(hmms, hmm_win_count, top_n=3):
    """Select the top HMMs based on win counts and KL divergence."""
    # Sort HMMs by win count
    sorted_indices = sorted(hmm_win_count, key=hmm_win_count.get, reverse=True)
    top_indices_by_win = sorted_indices[:top_n * 2]  # Take more for KL filtering
    
    # Filter top HMMs based on KL divergence
    kl_matrix = calculate_kl_divergence_matrix([hmms[i] for i in top_indices_by_win])
    avg_kl_divergence = np.mean(kl_matrix, axis=1)  # Average KL divergence for each HMM
    
    # Sort by average KL divergence to get the most distinct HMMs
    kl_sorted_indices = sorted(range(len(avg_kl_divergence)), key=lambda i: avg_kl_divergence[i], reverse=True)
    top_final_indices = [top_indices_by_win[i] for i in kl_sorted_indices[:top_n]]
    
    # Retrieve the final selected HMMs
    top_hmms = [hmms[i] for i in top_final_indices]
    print(f"Selected HMMs with highest win counts and distinctiveness: {top_final_indices}")
    return top_hmms


if __name__ == "__main__":
    from preprocessing import load_data, generate_sequences
    # from utils import save_hmms


    data_dir = 'C:\\Users\\Prakhar\\Desktop\\mtp\\final_implementation\\data'
    drillbit_data = load_data(data_dir)
    temporal_sequences = generate_sequences(drillbit_data)
    temporal_sequences_2d_list = generate_sequences_each_file(drillbit_data)
    
    # Initialize 10 HMMs with random subsets
    # hmms = initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2)
    hmms = initialize_hmms_limited_sequences(temporal_sequences_2d_list, num_hmms=10, num_sequences_per_hmm=10, num_iterations=2)
    
    # Perform competitive learning, ensuring sequence length is at least 4
    # hmms, top_hmms, hmm_win_count  = competitive_learning(temporal_sequences, hmms, max_epochs=20, min_sequence_length=4, tolerance=1e-4, max_refits=5)
    
    print("HMM training completed.")
