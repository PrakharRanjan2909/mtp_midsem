from preprocessing import load_data, generate_sequences, generate_sequences_each_file
from hmm_training import initialize_hmms, competitive_learning, initialize_hmms_limited_sequences
from utils import save_hmms
# from hmm_training import select_top_hmms_with_kl
# from kl_div_merging import merge_hmms

if __name__ == "__main__":
    data_dir = 'C:\\Users\\Prakhar\\Desktop\\mtp\\final_implementation\\data'

    # Step 1: Load and preprocess the data
    drillbit_data = load_data(data_dir)
    temporal_sequences = generate_sequences(drillbit_data)
    temporal_sequences_2d_list = generate_sequences_each_file(drillbit_data)
    print(f"Generated {len(temporal_sequences)} temporal sequences.")
    
    # # Step 2: Initialize and train 10 HMMs
    # hmms = initialize_hmms(temporal_sequences, num_hmms=10, num_iterations=2)
    hmms  = initialize_hmms_limited_sequences(temporal_sequences_2d_list, num_hmms=10, num_iterations=2)
    
     # Perform competitive learning, ensuring sequence length is at least 4
    hmms, top_hmms, hmm_win_count = competitive_learning(temporal_sequences, hmms, max_epochs=1, min_sequence_length=4, tolerance=1e-4, max_refits=5)
    

#   
  
    
    
    # Step 4: Save trained HMMs
    save_hmms(hmms, output_dir='./results/models/')

    # # Step 5: Evaluate and select the top 3 HMMs
    save_hmms(top_hmms, output_dir='./results/top_hmms/')
    # print("HMM training and selection completed.")
    # Step 5: Select the top 3 HMMs based on win count and KL divergence
    # top_hmms2 = select_top_hmms_with_kl(hmms, hmm_win_count, top_n=3)
    # Step 5: Merge 10 HMMs into 3 using KL divergence
    # merged_hmms = merge_hmms(hmms, target_num_hmms=3)

# Save merged HMMs
    # save_hmms(merged_hmms, output_dir='./results/merged_hmms/')

    # print("Merging of HMMs completed.")




# # Save selected HMMs
#     save_hmms(top_hmms2, output_dir='./results/top_hmms/')
#     print("Top HMMs selection completed.")


    

