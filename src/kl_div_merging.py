import numpy as np
from scipy.linalg import det, inv

from hmmlearn import hmm
import pickle
import os
from utils import save_hmms, load_hmms




# Function to calculate KL divergence between two Gaussian distributions
def kl_divergence_gaussians(mu1, cov1, mu2, cov2):
    cov2_inv = inv(cov2)
    term1 = np.log(det(cov2) / det(cov1))
    term2 = np.trace(cov2_inv @ cov1)
    term3 = (mu2 - mu1).T @ cov2_inv @ (mu2 - mu1)
    return 0.5 * (term1 + term2 + term3 - len(mu1))

# # Function to calculate pairwise KL divergence matrix for HMMs
# def calculate_kl_divergence_matrix(hmms):
#     n_hmms = len(hmms)
#     kl_matrix = np.zeros((n_hmms, n_hmms))
    
#     for i in range(n_hmms):
#         for j in range(i + 1, n_hmms):
#             kl_ij = 0
#             for state in range(hmms[i].n_components):
#                 mu_i = hmms[i].means_[state]
#                 cov_i = hmms[i].covars_[state]
#                 mu_j = hmms[j].means_[state]
#                 cov_j = hmms[j].covars_[state]
                
#                 kl_ij += kl_divergence_gaussians(mu_i, cov_i, mu_j, cov_j)
#             kl_ij /= hmms[i].n_components
#             kl_ji = kl_divergence_gaussians(mu_j, cov_j, mu_i, cov_i) / hmms[j].n_components
#             kl_matrix[i, j] = (kl_ij + kl_ji) / 2
#             kl_matrix[j, i] = kl_matrix[i, j]
    
#     return kl_matrix

def calculate_kl_divergence_matrix(hmms):
    """Calculate a pairwise KL divergence matrix for a list of HMMs."""
    n_hmms = len(hmms)
    kl_matrix = np.zeros((n_hmms, n_hmms))
    
    for i in range(n_hmms):
        for j in range(i + 1, n_hmms):
            kl_ij = 0
            
            # Get the number of features (dimensions) from the means shape
            n_features = hmms[i].means_.shape[1]
            
            # Calculate KL divergence for each state's Gaussian emission
            for state in range(hmms[i].n_components):
                mu_i = hmms[i].means_[state]
                cov_i = hmms[i].covars_[state].reshape(n_features, -1)  # Ensure proper shape
                mu_j = hmms[j].means_[state]
                cov_j = hmms[j].covars_[state].reshape(n_features, -1)  # Ensure proper shape
                
                kl_ij += kl_divergence_gaussians(mu_i, cov_i, mu_j, cov_j)
            
            # Average over states
            kl_ij /= hmms[i].n_components
            # Symmetrize the KL divergence by taking the average of both directions
            kl_matrix[i, j] = kl_matrix[j, i] = kl_ij
    
    return kl_matrix




def merge_hmms(hmms, target_num_hmms=3):
    """Iteratively merge HMMs using KL divergence until the target number of HMMs is reached."""
    while len(hmms) > target_num_hmms:
        # Calculate KL divergence matrix for the current set of HMMs
        kl_matrix = calculate_kl_divergence_matrix(hmms)
        
        # Find the indices of the two HMMs with the smallest KL divergence
        min_idx = np.unravel_index(np.argmin(kl_matrix + np.eye(len(kl_matrix)) * 1e6), kl_matrix.shape)
        i, j = min_idx
        print(f"Merging HMM {i} and HMM {j} with KL divergence {kl_matrix[i, j]}")

        # Merge HMM i and j by averaging their parameters
        merged_hmm = average_hmms(hmms[i], hmms[j])

        # Remove the merged HMMs and add the new merged HMM
        hmms = [hmms[k] for k in range(len(hmms)) if k not in min_idx]  # Remove i and j
        hmms.append(merged_hmm)
    
    return hmms

from hmmlearn import hmm

def average_hmms(hmm1, hmm2):
    """Average two HMMs to create a new merged HMM with diagonal covariances."""
    n_components = hmm1.n_components
    n_dim = hmm1.means_.shape[1]  # Dimensionality of the emissions

    # Create a new HMM model with the same structure
    merged_hmm = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

    # Average the transition matrices
    merged_hmm.transmat_ = (hmm1.transmat_ + hmm2.transmat_) / 2
    
    # Average the means
    merged_hmm.means_ = (hmm1.means_ + hmm2.means_) / 2

    # Access and average the diagonal covariances
    if hasattr(hmm1, '_covars_') and hasattr(hmm2, '_covars_'):
        # Use internal _covars_ attribute if available
        covars1 = hmm1._covars_
        covars2 = hmm2._covars_
    else:
        # Otherwise, use the public covars_ attribute
        covars1 = hmm1.covars_
        covars2 = hmm2.covars_
    
    # Ensure covars are shaped correctly and average them
    merged_hmm.covars_ = (covars1 + covars2) / 2

    return merged_hmm


# def average_hmms(hmm1, hmm2):
#     """Average two HMMs to create a new merged HMM."""
#     n_components = hmm1.n_components
#     merged_hmm = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")
    
#     # Average transition matrices
#     merged_hmm.transmat_ = (hmm1.transmat_ + hmm2.transmat_) / 2
    
#     # Average means and covariances for each state's Gaussian emission
#     merged_hmm.means_ = (hmm1.means_ + hmm2.means_) / 2
#     merged_hmm.covars_ = (hmm1.covars_ + hmm2.covars_) / 2
    
#     return merged_hmm

# def average_hmms(hmm1, hmm2):
#     """Average two HMMs to create a new merged HMM with diagonal covariances."""
#     n_components = hmm1.n_components
#     n_dim = hmm1.means_.shape[1]  # Dimensionality of the emissions

#     # Create a new HMM model with the same structure
#     merged_hmm = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

#     # Average the transition matrices
#     merged_hmm.transmat_ = (hmm1.transmat_ + hmm2.transmat_) / 2
    
#     # Average the means
#     merged_hmm.means_ = (hmm1.means_ + hmm2.means_) / 2

#     # Average the covariances in 'diag' format
#     # Ensure covariances are in shape (n_components, n_dim)
#     covars1 = hmm1.covars_.reshape(n_components, n_dim)
#     covars2 = hmm2.covars_.reshape(n_components, n_dim)
#     merged_hmm.covars_ = (covars1 + covars2) / 2

#     return merged_hmm

# def average_hmms(hmm1, hmm2):
#     """Average two HMMs to create a new merged HMM with diagonal covariances."""
#     n_components = hmm1.n_components
#     n_dim = hmm1.means_.shape[1]  # Dimensionality of the emissions

#     # Create a new HMM model with the same structure
#     merged_hmm = hmm.GaussianHMM(n_components=n_components, covariance_type="diag")

#     # Average the transition matrices
#     merged_hmm.transmat_ = (hmm1.transmat_ + hmm2.transmat_) / 2
    
#     # Average the means
#     merged_hmm.means_ = (hmm1.means_ + hmm2.means_) / 2

#     # Handle diagonal covariances: reshape and average
#     covars1 = hmm1.covars_.reshape(n_components, n_dim)
#     covars2 = hmm2.covars_.reshape(n_components, n_dim)
#     merged_covars = (covars1 + covars2) / 2
    
#     # Set the merged covariances correctly for 'diag' type
#     merged_hmm.covars_ = merged_covars

#     return merged_hmm




hmms = load_hmms('./results/models/')

merged_hmms = merge_hmms(hmms, target_num_hmms=3)

# Save merged HMMs
save_hmms(merged_hmms, output_dir='./results/merged_hmms/')

print("Merging of HMMs completed.")
