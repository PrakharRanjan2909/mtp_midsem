import numpy as np
from preprocessing import load_data, generate_sequences_each_file
from utils import load_hmms

def evaluate_hmm_wins(hmms, sequences):
    """
    Evaluate which HMM wins each sequence based on highest log-likelihood.
    """
    hmm_wins = []
    for seq in sequences:
        best_idx = None
        best_score = float('-inf')
        for idx, model in enumerate(hmms):
            score = model.score(seq)
            if score > best_score:
                best_score = score
                best_idx = idx
        hmm_wins.append(best_idx)
    return hmm_wins

def map_states_from_wins(hmm_wins):
    """
    Automatically assign good, mediocre, bad based on winning patterns.
    Handles small number of sequences safely.
    """
    n = len(hmm_wins)
    if n == 0:
        # No sequences at all
        return {}

    third = max(1, n // 3)  # Ensure at least 1

    good_section = hmm_wins[:third]
    mediocre_section = hmm_wins[third:2*third]
    bad_section = hmm_wins[2*third:]

    mapping = {}

    if good_section:
        good_hmm = max(set(good_section), key=good_section.count)
        mapping[good_hmm] = 'good'
    if mediocre_section:
        mediocre_hmm = max(set(mediocre_section), key=mediocre_section.count)
        mapping[mediocre_hmm] = 'mediocre'
    if bad_section:
        bad_hmm = max(set(bad_section), key=bad_section.count)
        mapping[bad_hmm] = 'bad'

    return mapping


def estimate_rul(hmm_wins, state_mapping, full_life_sequences=30):
    """
    Estimate remaining useful life (RUL) in terms of sequences or holes.
    """
    state_order = {'good': 0, 'mediocre': 1, 'bad': 2}
    health_sequence = [state_order[state_mapping[winner]] for winner in hmm_wins if winner in state_mapping]

    if not health_sequence:
        return 0

    # Find current health level based on last few sequences
    last_state = health_sequence[-1]

    # Estimate how much life is left based on health stage
    if last_state == 0:  # good
        # Assume good is first 1/3rd
        life_left_fraction = (2/3)
    elif last_state == 1:  # mediocre
        life_left_fraction = (1/3)
    else:  # bad
        life_left_fraction = (0)

    # RUL estimation (based on full expected sequences)
    rul_estimated = int(full_life_sequences * life_left_fraction)

    return max(rul_estimated, 0)

def main():
    data_dir = './new_data'  # Directory containing new unseen data
    hmms_dir = './results/top_hmms_kl/'  # Folder with selected top 3 HMMs

    # Load the new dataset
    drillbit_data = load_data(data_dir)
    sequences_2d = generate_sequences_each_file(drillbit_data)
    hmms = load_hmms(hmms_dir)

    for idx, sequences in enumerate(sequences_2d):
        print(f"\nProcessing Drill Bit {idx+1}:")

        hmm_wins = evaluate_hmm_wins(hmms, sequences)
        mapping = map_states_from_wins(hmm_wins)
        
        estimated_rul = estimate_rul(hmm_wins, mapping, full_life_sequences=30)  # you can adjust 30 if needed
        
        print(f"HMM Wins: {hmm_wins}")
        print(f"Health State Mapping: {mapping}")
        print(f"Estimated RUL (in sequences/holes): {estimated_rul}")

if __name__ == "__main__":
    main()
