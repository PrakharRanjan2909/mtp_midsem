# import numpy as np
# import pickle

# # Function to predict the best HMM for each sequence
# def predict_hmm(hmms, sequence):
#     """
#     Predict the HMM (health state) that best matches the given sequence.
#     """
#     best_hmm_idx = None
#     best_log_likelihood = float('-inf')
    
#     # Compare log-likelihood of the sequence for each HMM
#     for idx, model in enumerate(hmms):
#         log_likelihood = model.score(sequence)
#         if log_likelihood > best_log_likelihood:
#             best_log_likelihood = log_likelihood
#             best_hmm_idx = idx
    
#     return best_hmm_idx

# # Function to calculate accuracy for a dataset
# def calculate_accuracy(hmms, sequences, true_labels):
#     """
#     Calculate accuracy by comparing predicted HMMs with true labels.
#     """
#     correct_predictions = 0
#     total_predictions = len(sequences)
    
#     for i, seq in enumerate(sequences):
#         predicted_label = predict_hmm(hmms, seq)
#         if predicted_label == true_labels[i]:
#             correct_predictions += 1
    
#     accuracy = correct_predictions / total_predictions
#     return accuracy

# if __name__ == "__main__":
#     from preprocessing import load_data, generate_sequences
#     from utils import load_hmms
    
#     # Load top 3 HMMs for diagnostics
#     hmms = load_hmms(output_dir='./results/top_hmms/')
    
#     # Load the data
#     data_dir = '../data'
    
#     # Training set (first 10 drill bit files)
#     training_data = load_data(data_dir)[:10]
#     training_sequences = generate_sequences(training_data)
#     true_train_labels = [0] * len(training_sequences)  # Assume all training sequences are in "good" condition
    
#     # Test set (remaining files 11 to 14)
#     test_data = load_data(data_dir)[10:]
#     test_sequences = generate_sequences(test_data)
#     true_test_labels = [1] * len(test_sequences)  # Assume all test sequences are in "mediocre" condition
    
#     # Calculate training accuracy
#     train_accuracy = calculate_accuracy(hmms, training_sequences, true_train_labels)
#     print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    
#     # Calculate test accuracy
#     test_accuracy = calculate_accuracy(hmms, test_sequences, true_test_labels)
#     print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

import numpy as np
import pickle
from preprocessing import load_data, generate_sequences
from utils import load_hmms
import itertools


# Function to evaluate which HMM wins each sequence
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


# Function to map HMMs to health states based on winning pattern



def map_hmms_to_health_states(hmm_wins):
    n = len(hmm_wins)
    third = n // 3
    
    good_section = hmm_wins[:third]
    mediocre_section = hmm_wins[third:2*third]
    bad_section = hmm_wins[2*third:]
    
    good_hmm = max(set(good_section), key=good_section.count)
    mediocre_hmm = max(set(mediocre_section), key=mediocre_section.count)
    bad_hmm = max(set(bad_section), key=bad_section.count)
    
    return {good_hmm: 'good', mediocre_hmm: 'mediocre', bad_hmm: 'bad'}

# Function to calculate accuracy for a dataset
def calculate_accuracy(hmm_wins, health_state_mapping, alpha=0.05):
    state_sequence = [health_state_mapping[hmm] for hmm in hmm_wins]
    accuracy = 1.0  # Start with 100% accuracy
    
    for i in range(1, len(state_sequence)):
        if (state_sequence[i] == 'good' and state_sequence[i-1] in ['mediocre', 'bad']) or \
           (state_sequence[i] == 'mediocre' and state_sequence[i-1] == 'bad'):
            accuracy -= alpha  # Deduct alpha for each reverse jump
    
    return max(accuracy, 0) * 100  # Return as a percentage

# # Function to evaluate accuracy for training and test sets in one circular set
# def evaluate_set(hmms, datasets, health_state_mapping, train_indices, test_indices):
#     train_accuracies = []
#     test_accuracies = []
    
#     for idx in train_indices:
#         sequences = generate_sequences([datasets[idx]])
#         hmm_wins = evaluate_hmm_wins(hmms, sequences)
#         train_accuracies.append(calculate_accuracy(hmm_wins, health_state_mapping))
    
#     for idx in test_indices:
#         sequences = generate_sequences([datasets[idx]])
#         hmm_wins = evaluate_hmm_wins(hmms, sequences)
#         test_accuracies.append(calculate_accuracy(hmm_wins, health_state_mapping))
    
#     avg_train_accuracy = np.mean(train_accuracies)
#     avg_test_accuracy = np.mean(test_accuracies)
    
#     return avg_train_accuracy, avg_test_accuracy

# Function to evaluate accuracy for training and test sets in one circular set
def evaluate_set(hmms, datasets, health_state_mapping, train_indices, test_indices, alpha=0.05):
    train_accuracies = []
    test_accuracies = []
    
    for idx in train_indices:
        sequences = generate_sequences([datasets[idx]])
        hmm_wins = evaluate_hmm_wins(hmms, sequences)
        
        # Map the HMM win sequence to health states for better visualization
        
        state_sequence = [health_state_mapping [hmm] for hmm in hmm_wins]
        print(f"\nTraining Dataset {idx + 1} - Health State Sequence: {state_sequence}")
        
        # Calculate accuracy based on reverse jumps
        train_accuracy = calculate_accuracy(hmm_wins, health_state_mapping, alpha)
        train_accuracies.append(train_accuracy)
        print(f"Training Dataset {idx + 1} - Calculated Accuracy: {train_accuracy:.2f}%")
    
    for idx in test_indices:
        sequences = generate_sequences([datasets[idx]])
        hmm_wins = evaluate_hmm_wins(hmms, sequences)
        
        # Map the HMM win sequence to health states for better visualization
        state_sequence = [health_state_mapping[hmm] for hmm in hmm_wins]
        print(f"\nTest Dataset {idx + 1} - Health State Sequence: {state_sequence}")
        
        # Calculate accuracy based on reverse jumps
        test_accuracy = calculate_accuracy(hmm_wins, health_state_mapping, alpha)
        test_accuracies.append(test_accuracy)
        print(f"Test Dataset {idx + 1} - Calculated Accuracy: {test_accuracy:.2f}%")
    
    avg_train_accuracy = np.mean(train_accuracies)
    avg_test_accuracy = np.mean(test_accuracies)
    
    return avg_train_accuracy, avg_test_accuracy


# Function to perform circular cross-validation
def circular_cross_validation(hmms, datasets, num_splits=14, alpha=0.05):
    n = len(datasets)
    all_train_accuracies = []
    all_test_accuracies = []
    
    for split in range(num_splits):
        # Define train and test sets for this circular split
        test_indices = [(split + i) % n for i in range(n - 4, n)]  # 4 test datasets
        train_indices = [i for i in range(n) if i not in test_indices]  # Remaining datasets as training
        
        # Map HMMs to health states using the first training dataset in this split
        
        
        # print(test_indices)
        # print(datasets[train_indices[0]])
        sequences = generate_sequences([datasets[train_indices[0]]])
        hmm_wins = evaluate_hmm_wins(hmms, sequences)
        health_state_mapping = map_hmms_to_health_states(hmm_wins)
        
        # Evaluate training and test accuracy for this split
        avg_train_accuracy, avg_test_accuracy = evaluate_set(hmms, datasets, health_state_mapping, train_indices, test_indices)
        
        all_train_accuracies.append(avg_train_accuracy)
        all_test_accuracies.append(avg_test_accuracy)
        
        print(f"Set {split + 1}: Train Accuracy = {avg_train_accuracy:.2f}%, Test Accuracy = {avg_test_accuracy:.2f}%")
    
    # Calculate final average accuracies
    final_train_accuracy = np.mean(all_train_accuracies)
    final_test_accuracy = np.mean(all_test_accuracies)
    
    return final_train_accuracy, final_test_accuracy


# Main function to execute diagnostics
if __name__ == "__main__":
    data_dir = 'C:\\Users\\Prakhar\\Desktop\\mtp\\final_implementation\\data'
    datasets = load_data(data_dir)
   
    
    # Load the top 3 HMMs
    hmms = load_hmms(output_dir='C:\\Users\\Prakhar\\Desktop\\mtp\\final_implementation\\results\\top_hmms')
    
    # Perform circular cross-validation
    final_train_accuracy, final_test_accuracy = circular_cross_validation(hmms, datasets, num_splits=14, alpha=0.05)
    
    print(f"\nFinal Training Accuracy (averaged over all sets): {final_train_accuracy:.2f}%")




