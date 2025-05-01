import numpy as np
import pickle
import matplotlib.pyplot as plt
# Function to determine which HMM wins each sequence in a dataset
def evaluate_hmm_wins(hmms, sequences):
    """
    Evaluate which HMM wins each sequence by comparing log-likelihoods.
    Returns a list where each element corresponds to the index of the winning HMM for that sequence.
    """
    hmm_wins = []
    
    for seq in sequences:
        best_hmm_idx = None
        best_log_likelihood = float('-inf')
        
        for idx, model in enumerate(hmms):
            log_likelihood = model.score(seq)
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_hmm_idx = idx
        
        hmm_wins.append(best_hmm_idx)
    
    return hmm_wins

# Function to determine health state mapping (good, mediocre, bad)
def map_hmms_to_health_states(hmm_wins):
    """
    Map HMMs to health states based on the sequence winning pattern.
    Assumes that the first few sequences are in good condition, followed by mediocre, and finally bad.
    """
    # Assuming the progression from good -> mediocre -> bad based on sequence order
    n = len(hmm_wins)
    third = n // 5
    
    # Count the HMM wins in three sections (beginning -> middle -> end)
    good_section = hmm_wins[:third]
    mediocre_section = hmm_wins[third:2*third]
    bad_section = hmm_wins[2*third:]
    
    # Count which HMM wins the most in each section
    good_hmm = max(set(good_section), key=good_section.count)
    mediocre_hmm = max(set(mediocre_section), key=mediocre_section.count)
    bad_hmm = max(set(bad_section), key=bad_section.count)
    
    return {good_hmm: 'good', mediocre_hmm: 'mediocre', bad_hmm: 'bad'}


# Function to evaluate log-likelihoods for each HMM over sequences
def evaluate_hmm_log_likelihoods(hmms, sequences):
    """
    Evaluate log-likelihood for each HMM over a set of sequences.
    Returns a 2D array where each row corresponds to an HMM and each column corresponds to a sequence.
    """
    log_likelihoods = np.zeros((len(hmms), len(sequences)))
    
    for i, seq in enumerate(sequences):
        for j, model in enumerate(hmms):
            log_likelihoods[j, i] = model.score(seq)
    
    return log_likelihoods

# Function to plot log-likelihoods for each HMM over the sequences
def plot_log_likelihoods(log_likelihoods):
    """
    Plot the log-likelihoods of each HMM over the temporal sequences (holes).
    """
    num_sequences = log_likelihoods.shape[1]
    
    # Plot each HMM's log-likelihoods as a separate line
    for hmm_idx in range(log_likelihoods.shape[0]):
        plt.plot(range(num_sequences), log_likelihoods[hmm_idx], label=f'HMM {hmm_idx+1}', marker='o', linestyle = 'solid')
    
    plt.title('Log-Likelihood vs Temporal Sequences (Holes)')
    plt.xlabel('Sequence (Hole Index)')
    plt.ylabel('Log-Likelihood')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    from preprocessing import load_data, generate_sequences
    from utils import load_hmms
    
    # Load the top 3 HMMs
    hmms = load_hmms(output_dir='./results/top_hmms/')
    
    # Load the dataset (for example DB1.txt)
    data_dir = 'C:\\Users\\Prakhar\\Desktop\\mtp\\final_implementation\\data'
    drillbit_data = load_data(data_dir)
    
    # Evaluate which HMM wins each sequence for the first drill bit (DB1.txt)
    x = 7
    sequences_db1 = generate_sequences([drillbit_data[x]])  # Assuming DB1.txt is the first file
    hmm_wins_db1 = evaluate_hmm_wins(hmms, sequences_db1)
    
    # Print the sequence of winning HMMs for Drill Bit 1
    print(f"Winning HMMs for each sequence in DB: {hmm_wins_db1}")
    
    # Map the HMMs to health states (good, mediocre, bad) based on the winning pattern
    health_state_mapping = map_hmms_to_health_states(hmm_wins_db1)
    print(f"HMM to health state mapping: 1: good, 0: mediocre, 2: bad")


     # Compute log-likelihoods for each HMM over the sequences
    log_likelihoods = evaluate_hmm_log_likelihoods(hmms, sequences_db1)
    
    # Plot log-likelihoods for all HMMs over the sequences
    plot_log_likelihoods(log_likelihoods)
