import numpy as np

def simulate_rul_monte_carlo(hmms, current_state, health_state_mapping, num_simulations=1000):
    """
    Estimate Remaining Useful Life (RUL) using Monte Carlo simulation from the current state.
    This function runs multiple simulations to predict the number of steps until reaching the 'bad' state.
    
    Parameters:
        hmms (list): List of trained HMMs.
        current_state (str): The current health state of the tool ('good', 'mediocre', or 'bad').
        health_state_mapping (dict): Mapping of HMM indices to health states.
        num_simulations (int): Number of Monte Carlo simulations to run.
        
    Returns:
        float: Estimated RUL in terms of the expected number of sequences (holes) remaining.
    """
    state_to_hmm = {v: k for k, v in health_state_mapping.items()}
    good_hmm = state_to_hmm['good']
    mediocre_hmm = state_to_hmm['mediocre']
    bad_hmm = state_to_hmm['bad']
    
    # Initial HMM based on the current state
    current_hmm_idx = state_to_hmm[current_state]
    current_hmm = hmms[current_hmm_idx]

    # Monte Carlo simulation to estimate RUL
    rul_estimates = []
    for _ in range(num_simulations):
        steps = 0
        while current_hmm_idx != bad_hmm:
            transmat = current_hmm.transmat_
            next_state = np.random.choice([good_hmm, mediocre_hmm, bad_hmm], p=transmat[current_hmm_idx])
            steps += 1
            current_hmm_idx = next_state
            if current_hmm_idx == bad_hmm:
                break
            current_hmm = hmms[current_hmm_idx]
        rul_estimates.append(steps)
    
    # Average RUL from simulations
    average_rul = np.mean(rul_estimates)
    std_rul = np.std(rul_estimates)  # Optional: to quantify uncertainty

    return average_rul, std_rul

# Example usage:
if __name__ == "__main__":
    from utils import load_hmms
    
    # Load HMMs and current state
    hmms = load_hmms(output_dir='./results/top_hmms/')
    health_state_mapping = {0: 'good', 1: 'mediocre', 2: 'bad'}  # Example mapping
    
    # Assume the tool is currently in a 'mediocre' state
    current_state = 'mediocre'
    
    # Estimate RUL
    rul_estimate, rul_std = simulate_rul_monte_carlo(hmms, current_state, health_state_mapping)
    print(f"Estimated RUL (Monte Carlo Simulation-Based): {rul_estimate:.2f} sequences ± {rul_std:.2f} sequences")
