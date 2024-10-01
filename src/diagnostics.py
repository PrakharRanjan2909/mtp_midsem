import numpy as np
import pickle

# Function to predict the best HMM for each sequence
def predict_hmm(hmms, sequence):
    """
    Predict the HMM (health state) that best matches the given sequence.
    """
    best_hmm_idx = None
    best_log_likelihood = float('-inf')
    
    # Compare log-likelihood of the sequence for each HMM
    for idx, model in enumerate(hmms):
        log_likelihood = model.score(sequence)
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_hmm_idx = idx
    
    return best_hmm_idx

# Function to calculate accuracy for a dataset
def calculate_accuracy(hmms, sequences, true_labels):
    """
    Calculate accuracy by comparing predicted HMMs with true labels.
    """
    correct_predictions = 0
    total_predictions = len(sequences)
    
    for i, seq in enumerate(sequences):
        predicted_label = predict_hmm(hmms, seq)
        if predicted_label == true_labels[i]:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

if __name__ == "__main__":
    from preprocessing import load_data, generate_sequences
    from utils import load_hmms
    
    # Load top 3 HMMs for diagnostics
    hmms = load_hmms(output_dir='./results/top_hmms/')
    
    # Load the data
    data_dir = '../data'
    
    # Training set (first 10 drill bit files)
    training_data = load_data(data_dir)[:10]
    training_sequences = generate_sequences(training_data)
    true_train_labels = [0] * len(training_sequences)  # Assume all training sequences are in "good" condition
    
    # Test set (remaining files 11 to 14)
    test_data = load_data(data_dir)[10:]
    test_sequences = generate_sequences(test_data)
    true_test_labels = [1] * len(test_sequences)  # Assume all test sequences are in "mediocre" condition
    
    # Calculate training accuracy
    train_accuracy = calculate_accuracy(hmms, training_sequences, true_train_labels)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    
    # Calculate test accuracy
    test_accuracy = calculate_accuracy(hmms, test_sequences, true_test_labels)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
