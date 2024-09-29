# # import numpy as np
# # from hmmlearn import hmm
# # import random

# # # Function to initialize HMMs with random subsets
# # def initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2):
# #     """
# #     Initialize HMMs by training them with random subsets of the temporal sequences
# #     for a few iterations (e.g., 2 iterations).
# #     """
# #     # if not temporal_sequences:
# #     #     raise ValueError("No temporal sequences provided for HMM initialization.")
# #     hmms = []
    
# #     # Initialize HMMs
# #     for i in range(num_hmms):
# #         model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=num_iterations)
        
# #         # Randomly select a subset of sequences to train the HMM initially
# #         print(f"sahpe of temporal_sequences: {len(temporal_sequences)}")
# #         random_subset = random.sample(temporal_sequences, 5)
# #         # random_subset = np.random.choice(temporal_sequences, size=5, replace=False)
# #         print(f"Training HMM {i} with {len(random_subset)} sequences")
# #         for(i, seq) in enumerate(random_subset):
# #             print(f"Sequence {i} shape: {seq.shape}")
# #         lengths = [len(seq) for seq in random_subset]
# #         combined_data = np.vstack(random_subset)  # Combine all sequences into one for training
        
# #         # Train the HMM on this random subset for a couple of iterations
# #         model.fit(combined_data, lengths)
# #         hmms.append(model)
        
# #     print(f"Trained HMM")
# #     return hmms

# # # Function for competitive learning using HMMs
# # def competitive_learning(temporal_sequences, hmms, max_epochs=10, min_sequence_length=4, tolerance=1e-4):
# #     """
# #     Perform competitive learning on the temporal sequences with the initialized HMMs.
# #     Each sequence is assigned to the HMM that maximizes the log-likelihood, and the
# #     winning HMM updates its parameters.
# #     """
# #     dict = {}
# #     for epoch in range(max_epochs):
        
# #         print(f"Epoch {epoch + 1}/{max_epochs}")
# #         print(f"Competitive Training HMMs with {len(temporal_sequences)} sequences")
# #         # Track total log-likelihood change to check for convergence
# #         total_log_likelihood_change = 0

# #         for seq in temporal_sequences:
# #             seq = np.array(seq)  # Ensure the sequence is a numpy array

# #             # Skip sequences that are too short for the number of HMM states
# #             if len(seq) < min_sequence_length:
# #                 print(f"Skipping sequence of length {len(seq)} (too short for HMM with {min_sequence_length} states)")
# #                 continue


# #             best_hmm = None
# #             best_log_likelihood = float('-inf')
# #             best_idx = -1
# #             #print hmms shape
# #             print(f"hmms shape: {len(hmms)}")
# #             # Evaluate each HMM on the sequence
# #             for idx, model in enumerate(hmms):
# #                 log_likelihood = model.score(seq)  # Calculate log-likelihood for the sequence
# #                 if log_likelihood > best_log_likelihood:
# #                     best_log_likelihood = log_likelihood
# #                     print(f"Best log likelihood: {best_log_likelihood}")
# #                     best_hmm = model
# #                     print(f"Best hmm: {best_hmm}")
# #                     best_idx = idx
# #                     print(f"Best idx: {best_idx}")
            
# #             # Combine the current sequence into a 2D array (fit expects 2D)
# #             seq = seq.reshape(-1, 2)  # Reshape to (n_samples, n_features)


# #             # Save the previous parameters to check for convergence
# #             old_params = best_hmm.transmat_.copy()

# #             # # Update the winning HMM with the sequence
# #             best_hmm.fit(seq, [len(seq)])
# #             print(f"HMM {best_idx} won the sequence")

# #             if best_idx in dict:
# #                 dict[best_idx] += 1
# #             else:
# #                 dict[best_idx] = 1
# #             for key in dict:
# #                 print(f"Key: {key}, Value: {dict[key]}")
            
# #             # # Ensure the sequence has enough samples to fit the HMM
# #             # if len(seq) >= best_hmm.n_components:
# #             #     best_hmm.fit(seq, [len(seq)])
# #             #     print(f"HMM {best_idx} won the sequence")
# #             # else:
# #             #     print(f"Sequence too short for HMM {best_idx}: {len(seq)} samples, {best_hmm.n_components} components")
# #             # Compute log-likelihood change to track improvement
# #             new_log_likelihood = best_hmm.score(seq)
# #             likelihood_change = np.abs(new_log_likelihood - best_log_likelihood)
# #             total_log_likelihood_change += likelihood_change
            
# #             # Check if the HMM has converged based on parameter change or log-likelihood change
# #             if np.max(np.abs(best_hmm.transmat_ - old_params)) < tolerance:
# #                 print(f"HMM {best_idx} has converged on this sequence.")
# #                 break
        
# #         print(f"Total log-likelihood change in epoch {epoch + 1}: {total_log_likelihood_change}")
        
# #         # Early stopping criterion based on total log-likelihood change across the epoch
# #         if total_log_likelihood_change < tolerance:
# #             print(f"Convergence reached after {epoch + 1} epochs.")
# #             break
        
# #     print(f"Competitive learning completed.")
# #     #print the dict
    
    
# #     return hmms

# # if __name__ == "__main__":
# #     from preprocessing import load_data, generate_sequences
    
# #     data_dir = '../data'
# #     drillbit_data = load_data(data_dir)
# #     # temporal_sequences = [np.random.rand(np.random.randint(5, 15), 2) for _ in range(441)]
# #     temporal_sequences = generate_sequences(drillbit_data)
    
# #     # Initialize 10 HMMs with random subsets
# #     hmms = initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2)
    
# #     # Perform competitive learning
# #     hmms = competitive_learning(temporal_sequences, hmms, max_epochs=10, min_sequence_length=4, tolerance=1e-4)
    
# #     print("HMM training completed.")
# import random
# import numpy as np
# from hmmlearn import hmm

# # Function to initialize HMMs with random subsets
# def initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2):
#     """
#     Initialize HMMs by training them with random subsets of the temporal sequences
#     for a few iterations (e.g., 2 iterations).
#     """
#     hmms = []
    
#     # Initialize HMMs
#     for i in range(num_hmms):
#         model = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=num_iterations)
        
#         # Randomly select a subset of sequences to train the HMM initially
#         random_subset = random.sample(temporal_sequences, 5)
#         lengths = [len(seq) for seq in random_subset]
#         combined_data = np.vstack(random_subset)  # Combine all sequences into one for training
        
#         # Train the HMM on this random subset for a couple of iterations
#         model.fit(combined_data, lengths)
#         hmms.append(model)
    
#     return hmms

# # Function for competitive learning using HMMs with a stopping criterion
# def competitive_learning(temporal_sequences, hmms, max_epochs=2, min_sequence_length=4, tolerance=1e-4):
#     """
#     Perform competitive learning on the temporal sequences with the initialized HMMs.
#     Each sequence is assigned to the HMM that maximizes the log-likelihood, and the
#     winning HMM updates its parameters.
    
#     Only sequences that are at least as long as the number of HMM states are used for training.
#     Includes convergence checks based on log-likelihood improvements.
#     """
#     for epoch in range(max_epochs):
#         print(f"Epoch {epoch + 1}/{max_epochs}")
        
#         # Track total log-likelihood change to check for convergence
#         total_log_likelihood_change = 0
        
#         for seq in temporal_sequences:
#             seq = np.array(seq)  # Ensure the sequence is a numpy array
            
#             # Skip sequences that are too short for the number of HMM states
#             if len(seq) < min_sequence_length:
#                 print(f"Skipping sequence of length {len(seq)} (too short for HMM with {min_sequence_length} states)")
#                 continue
            
#             best_hmm = None
#             best_log_likelihood = float('-inf')
#             best_idx = -1
            
#             # Evaluate each HMM on the sequence
#             for idx, model in enumerate(hmms):
#                 log_likelihood = model.score(seq)  # Calculate log-likelihood for the sequence
#                 if log_likelihood > best_log_likelihood:
#                     best_log_likelihood = log_likelihood
#                     best_hmm = model
#                     best_idx = idx
            
#             # Combine the current sequence into a 2D array (fit expects 2D)
#             seq = seq.reshape(-1, 2)  # Reshape to (n_samples, n_features)
            
#             # Save the previous parameters to check for convergence
#             old_params = best_hmm.transmat_.copy()
            
#             # Update the winning HMM with the sequence
#             best_hmm.fit(seq, [len(seq)])  # Fit using the sequence and its length
#             print(f"HMM {best_idx} won the sequence")
            
#             # Compute log-likelihood change to track improvement
#             new_log_likelihood = best_hmm.score(seq)
#             likelihood_change = np.abs(new_log_likelihood - best_log_likelihood)
#             total_log_likelihood_change += likelihood_change
            
#             # Check if the HMM has converged based on parameter change or log-likelihood change
#             if np.max(np.abs(best_hmm.transmat_ - old_params)) < tolerance:
#                 print(f"HMM {best_idx} has converged on this sequence.")
#                 break
        
#         print(f"Total log-likelihood change in epoch {epoch + 1}: {total_log_likelihood_change}")
        
#         # Early stopping criterion based on total log-likelihood change across the epoch
#         if total_log_likelihood_change < tolerance:
#             print(f"Convergence reached after {epoch + 1} epochs.")
#             break
    
#     return hmms

# if __name__ == "__main__":
#     from preprocessing import load_data, generate_sequences
    
#     data_dir = '../data'
#     drillbit_data = load_data(data_dir)
#     temporal_sequences = generate_sequences(drillbit_data)
    
#     # Initialize 10 HMMs with random subsets
#     hmms = initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2)
    
#     # Perform competitive learning, ensuring sequence length is at least 4
#     hmms = competitive_learning(temporal_sequences, hmms, max_epochs=10, min_sequence_length=4, tolerance=1e-4)
    
#     print("HMM training completed.")
import random
import numpy as np
from hmmlearn import hmm

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

# Function for competitive learning using HMMs with a stopping criterion
def competitive_learning(temporal_sequences, hmms, max_epochs=10, min_sequence_length=4, tolerance=1e-4, max_refits=5):
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
    
    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        
        # Track total log-likelihood change to check for convergence
        total_log_likelihood_change = 0
        
        for idx, seq in enumerate(temporal_sequences):
            seq = np.array(seq)  # Ensure the sequence is a numpy array
            
            # Skip sequences that are too short for the number of HMM states
            if len(seq) < min_sequence_length:
                print(f"Skipping sequence of length {len(seq)} (too short for HMM with {min_sequence_length} states)")
                continue
            
            # Skip refitting if a sequence has been refit too many times
            if refit_counts[idx] >= max_refits:
                print(f"Skipping sequence {idx} as it has been refit {max_refits} times")
                continue
            
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
            best_hmm.fit(seq, [len(seq)])  # Fit using the sequence and its length
            refit_counts[idx] += 1  # Increment refit counter for the sequence
            
            print(f"HMM {best_hmm_idx} won the sequence {idx}")

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
        
        # Early stopping criterion based on total log-likelihood change across the epoch
        if total_log_likelihood_change < tolerance:
            print(f"Convergence reached after {epoch + 1} epochs.")
            break
    
    return hmms

if __name__ == "__main__":
    from preprocessing import load_data, generate_sequences
    
    data_dir = '../data'
    drillbit_data = load_data(data_dir)
    temporal_sequences = generate_sequences(drillbit_data)
    
    # Initialize 10 HMMs with random subsets
    hmms = initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2)
    
    # Perform competitive learning, ensuring sequence length is at least 4
    hmms = competitive_learning(temporal_sequences, hmms, max_epochs=10, min_sequence_length=4, tolerance=1e-4, max_refits=5)
    
    print("HMM training completed.")
