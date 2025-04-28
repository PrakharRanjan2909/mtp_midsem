# import numpy as np

# def estimate_rul_transition_matrix(hmms, current_state, health_state_mapping):
#     """
#     Estimate Remaining Useful Life (RUL) using the transition matrix from the current state.
#     This function calculates the expected number of transitions to reach the 'bad' state.
    
#     Parameters:
#         hmms (list): List of trained HMMs.
#         current_state (str): The current health state of the tool ('good', 'mediocre', or 'bad').
#         health_state_mapping (dict): Mapping of HMM indices to health states.
        
#     Returns:
#         float: Estimated RUL in terms of the expected number of sequences (holes) remaining.
#     """
#     # Identify which HMM represents each health state
#     state_to_hmm = {v: k for k, v in health_state_mapping.items()}
#     good_hmm = state_to_hmm['good']
#     mediocre_hmm = state_to_hmm['mediocre']
#     bad_hmm = state_to_hmm['bad']

#     # Get the current HMM based on the current health state
#     current_hmm_idx = state_to_hmm[current_state]
#     current_hmm = hmms[current_hmm_idx]
    
#     # Get the transition matrix of the current HMM
#     transmat = current_hmm.transmat_

#     # Initialize expected transitions to reach 'bad'
#     expected_steps_to_bad = 0.0
#     current_prob = 1.0  # Starting probability of being in the current state

#     # Traverse through states until reaching 'bad'
#     while current_hmm_idx != bad_hmm:
#         # Probability of transitioning to 'bad' state in the next step
#         prob_to_bad = transmat[current_hmm_idx, bad_hmm]
        
#         # Expected steps to reach 'bad' (inverse probability of transitioning to 'bad')
#         if prob_to_bad > 0:
#             expected_steps_to_bad += current_prob / prob_to_bad
#             break
#         else:
#             # Sum up expected steps within the current state
#             stay_prob = transmat[current_hmm_idx, current_hmm_idx]
#             expected_steps_to_bad += current_prob * (1 / (1 - stay_prob))
#             current_prob *= stay_prob

#             # Move to the next most probable degraded state
#             if current_hmm_idx == good_hmm:
#                 current_hmm_idx = mediocre_hmm
#             elif current_hmm_idx == mediocre_hmm:
#                 current_hmm_idx = bad_hmm

#     return expected_steps_to_bad

# # Example usage:
# if __name__ == "__main__":
#     from utils import load_hmms
    
#     # Load HMMs and current state
#     hmms = load_hmms(output_dir='./results/top_hmms/')
#     health_state_mapping = {0: 'good', 1: 'mediocre', 2: 'bad'}  # Example mapping
    
#     # Assume the tool is currently in a 'mediocre' state
#     current_state = 'mediocre'
    
#     # Estimate RUL
#     rul_estimate = estimate_rul_transition_matrix(hmms, current_state, health_state_mapping)
#     print(f"Estimated RUL (Transition Matrix-Based): {rul_estimate:.2f} sequences")
import numpy as np
from preprocessing import generate_sequences
from utils import load_hmms
from diagnostics import evaluate_hmm_wins, map_hmms_to_health_states
from hmmlearn import hmm

# Step 1: Preprocess and Load New Dataset
def preprocess_new_dataset(new_data_file):
    """
    Preprocess the new dataset to generate temporal sequences.
    
    Parameters:
        new_data_file (str): Path to the new dataset file.
        
    Returns:
        list: List of temporal sequences from the new dataset.
    """
    # Load and preprocess the dataset
    data = np.loadtxt(new_data_file)  # Assuming data is in a similar format as other datasets
    temporal_sequences = generate_sequences([data])
    return temporal_sequences

# Step 2: Determine the Current Tool State
def determine_current_state(hmms, sequences, health_state_mapping):
    """
    Determine the current health state of the tool based on the most frequent winning HMM.
    
    Parameters:
        hmms (list): List of trained HMMs.
        sequences (list): Temporal sequences from the new dataset.
        health_state_mapping (dict): Mapping of HMM indices to health states.
        
    Returns:
        str: The current health state of the tool ('good', 'mediocre', or 'bad').
    """
    # Evaluate which HMM wins each sequence
    hmm_wins = evaluate_hmm_wins(hmms, sequences)
    
    # Determine the most frequent state based on wins
    state_sequence = [health_state_mapping[hmm] for hmm in hmm_wins]
    current_state = max(set(state_sequence), key=state_sequence.count)
    
    print(f"Current tool state determined as: {current_state}")
    return current_state

# Step 3a: RUL Estimation using Transition Matrix-Based Method
def estimate_rul_transition_matrix(hmms, current_state, health_state_mapping):
    state_to_hmm = {v: k for k, v in health_state_mapping.items()}
    good_hmm = state_to_hmm['good']
    mediocre_hmm = state_to_hmm['mediocre']
    bad_hmm = state_to_hmm['bad']
    
    current_hmm_idx = state_to_hmm[current_state]
    current_hmm = hmms[current_hmm_idx]
    
    transmat = current_hmm.transmat_
    expected_steps_to_bad = 0.0
    current_prob = 1.0  # Starting probability of being in the current state

    while current_hmm_idx != bad_hmm:
        prob_to_bad = transmat[current_hmm_idx, bad_hmm]
        
        if prob_to_bad > 0:
            expected_steps_to_bad += current_prob / prob_to_bad
            break
        else:
            stay_prob = transmat[current_hmm_idx, current_hmm_idx]
            expected_steps_to_bad += current_prob * (1 / (1 - stay_prob))
            current_prob *= stay_prob

            if current_hmm_idx == good_hmm:
                current_hmm_idx = mediocre_hmm
            elif current_hmm_idx == mediocre_hmm:
                current_hmm_idx = bad_hmm

    return expected_steps_to_bad


if __name__ == "__main__":
    new_data_file = './data/DB7.txt'  # Example path to the new dataset
    
    # Load HMMs and health state mapping
    
    hmms = load_hmms(output_dir='./results/top_hmms/')
    health_state_mapping = {0: 'good', 1: 'mediocre', 2: 'bad'}  # Example mapping

    # Step 1: Preprocess the new dataset
    sequences = preprocess_new_dataset(new_data_file)
    
    # Step 2: Determine the current health state of the tool
    current_state = determine_current_state(hmms, sequences, health_state_mapping)
    
    # Step 3a: Estimate RUL using Transition Matrix-Based method
    rul_transition = estimate_rul_transition_matrix(hmms, current_state, health_state_mapping)
    print(f"Estimated RUL (Transition Matrix-Based): {rul_transition:.2f} sequences")

    # # Step 3b: Estimate RUL using Monte Carlo Simulation
    # rul_monte_carlo, rul_std = simulate_rul_monte_carlo(hmms, current_state, health_state_mapping)
    # print(f"Estimated RUL (Monte Carlo Simulation): {rul_monte_carlo:.2f} sequences ± {rul_std:.2f} sequences")
