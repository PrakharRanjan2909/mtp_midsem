import os
import pickle 

# Utility functions for the project can go here, such as:
# - Loading files
# - Splitting sequences
# - Log-likelihood calculations, etc.

def save_hmms(hmms, output_dir='../results/models/'):
    """
    Save trained HMM models to disk for later use.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, model in enumerate(hmms):
        model_file = os.path.join(output_dir, f"hmm_model_{i}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"Saved HMM {i} to {model_file}")

def load_hmms(output_dir='../results/models/'):
    """
    Load previously trained HMM models from disk.
    """
    hmms = []
    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".pkl"):
            model_file = os.path.join(output_dir, filename)
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
                hmms.append(model)
    return hmms
