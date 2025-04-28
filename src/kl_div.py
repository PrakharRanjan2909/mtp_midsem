import os
import numpy as np
import pickle
from tqdm import tqdm

def simulate_sequences(hmm_model, num_sequences=20, seq_length=50):
    sequences = []
    for _ in range(num_sequences):
        X, _ = hmm_model.sample(seq_length)
        sequences.append(X)
    return sequences

def approximate_kl_divergence(hmm_p, hmm_q, num_sequences=20, seq_length=50):
    sequences = simulate_sequences(hmm_p, num_sequences, seq_length)
    kl = 0.0
    for seq in sequences:
        log_p = hmm_p.score(seq)
        log_q = hmm_q.score(seq)
        kl += (log_p - log_q)
    return kl / num_sequences

def compute_kl_matrix(hmms, num_sequences=20, seq_length=50):
    n = len(hmms)
    kl_matrix = np.zeros((n, n))
    for i in tqdm(range(n), desc="Computing KL Divergence Matrix"):
        for j in range(n):
            if i != j:
                kl_matrix[i, j] = approximate_kl_divergence(hmms[i], hmms[j], num_sequences, seq_length)
            else:
                kl_matrix[i, j] = 0.0
    return kl_matrix

def select_most_diverse_hmms(kl_matrix, top_n=3):
    n = kl_matrix.shape[0]
    selected = []

    # Start with HMM having maximum total divergence
    first = np.argmax(np.sum(kl_matrix, axis=1))
    selected.append(first)

    for _ in range(top_n - 1):
        remaining = [i for i in range(n) if i not in selected]
        scores = []
        for r in remaining:
            score = sum(kl_matrix[r, s] + kl_matrix[s, r] for s in selected)
            scores.append(score)
        next_hmm = remaining[np.argmax(scores)]
        selected.append(next_hmm)

    return selected

def save_selected_hmms(hmms, selected_indices, output_dir='./results/top_hmms_kl/'):
    """
    Save the selected HMMs into a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    for idx in selected_indices:
        filename = os.path.join(output_dir, f'hmm_{idx}.pkl')
        with open(filename, 'wb') as f:
            pickle.dump(hmms[idx], f)
    print(f"Saved selected HMMs into {output_dir}")

from utils import load_hmms
hmms = load_hmms('./results/models/')  # Load all 10 trained HMMs

kl_matrix = compute_kl_matrix(hmms)
top_hmms_idx = select_most_diverse_hmms(kl_matrix, top_n=3)
save_selected_hmms(hmms, top_hmms_idx, output_dir='./results/top_hmms_kl/')
