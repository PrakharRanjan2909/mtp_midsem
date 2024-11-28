import numpy as np
from scipy.linalg import det, inv

# Function to calculate KL divergence between two Gaussians
def kl_divergence_gaussians(mu1, cov1, mu2, cov2):
    """Calculate KL divergence between two Gaussian distributions."""
    cov2_inv = inv(cov2)
    term1 = np.log(det(cov2) / det(cov1))
    term2 = np.trace(cov2_inv @ cov1)
    term3 = (mu2 - mu1).T @ cov2_inv @ (mu2 - mu1)
    return 0.5 * (term1 + term2 + term3 - len(mu1))

# Function to calculate pairwise KL divergence between HMMs
def calculate_kl_divergence_matrix(hmms):
    """Calculate a pairwise KL divergence matrix for a list of HMMs."""
    n_hmms = len(hmms)
    kl_matrix = np.zeros((n_hmms, n_hmms))
    
    for i in range(n_hmms):
        for j in range(i + 1, n_hmms):
            kl_ij = 0
            for state in range(hmms[i].n_components):
                mu_i = hmms[i].means_[state]
                cov_i = hmms[i].covars_[state]
                mu_j = hmms[j].means_[state]
                cov_j = hmms[j].covars_[state]
                
                # KL divergence for each state's Gaussian emission
                kl_ij += kl_divergence_gaussians(mu_i, cov_i, mu_j, cov_j)
            
            # Average over states
            kl_ij /= hmms[i].n_components
            # Symmetrize the KL divergence
            kl_ji = kl_divergence_gaussians(mu_j, cov_j, mu_i, cov_i) / hmms[j].n_components
            kl_matrix[i, j] = (kl_ij + kl_ji) / 2
            kl_matrix[j, i] = kl_matrix[i, j]
            
    return kl_matrix