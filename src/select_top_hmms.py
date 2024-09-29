import os
import numpy as np
import pickle

# Function to score HMMs based on average log-likelihood over training sequences
def rank_hmms(hmms, sequences):
    """
    Rank HMMs based on their average log-likelihood score over a set of sequences.
    Returns the ranked HMMs and their average log-likelihoods.
    """
    scores = []
    
    for model in hmms:
        log_likelihoods = []
        for seq in sequences:
            log_likelihoods.append(model.score(seq))
        avg_log_likelihood = np.mean(log_likelihoods)
        scores.append(avg_log_likelihood)
    
    # Rank HMMs by average log-likelihood
    ranked_indices = np.argsort(scores)[::-1]  # Sort in descending order
    ranked_hmms = [hmms[i] for i in ranked_indices[:3]]  # Top 3 HMMs
    ranked_scores = [scores[i] for i in ranked_indices[:3]]
    
    return ranked_hmms, ranked_scores

# Function to save the top 3 HMMs to a new folder
def save_top_hmms(hmms, output_dir='./results/top_hmms/'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, model in enumerate(hmms):
        model_file = os.path.join(output_dir, f"top_hmm_{i}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved HMM {i} to {model_file}")

if __name__ == "__main__":
    from preprocessing import load_data, generate_sequences
    from utils import load_hmms
    
    # Load the data and the previously trained HMMs
    data_dir = '../data'
    drillbit_data = load_data(data_dir)
    temporal_sequences = generate_sequences(drillbit_data[:10])  # Use the first 10 files as training set
    
    # Load the trained HMMs from the results folder
    hmms = load_hmms(output_dir='../results/models/')
    
    # Rank the HMMs based on their average log-likelihood over the training sequences
    top_hmms, top_scores = rank_hmms(hmms, temporal_sequences)
    
    print(f"Top 3 HMMs based on average log-likelihood: {top_scores}")
    
    # Save the top 3 HMMs to a separate folder
    save_top_hmms(top_hmms, output_dir='./results/top_hmms/')
