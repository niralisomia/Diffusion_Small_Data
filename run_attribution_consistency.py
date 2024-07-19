import numpy as np
import pickle
from datetime import datetime

from PL_DeepSTARR import *
from seq_evals_improved import *
from helpers import *

if __name__=='__main__':
    
    # get the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # loading data
    samples = np.load("samples.npz")
    sample_seqs = samples['arr_0']
    sample_seqs = torch.tensor(sample_seqs, dtype=torch.float32) # suppossed to be (N, L, A)

    DeepSTARR_data = h5py.File("/grid/koo/home/ykang/d3_test/DeepSTARR_data.h5", 'r')
    X_test = torch.tensor(np.array(DeepSTARR_data['X_test']).transpose(0,2,1), dtype=torch.float32)


    deepstarr = PL_DeepSTARR.load_from_checkpoint("/grid/koo/home/ykang/d3_test/oracle_DeepSTARR_DeepSTARR_data.ckpt", 
                                                  input_h5_file = "/grid/koo/home/ykang/d3_test/DeepSTARR_data.h5").eval()
    
    # top 2,000 functional activity sampled sequence
    ## call deepstarr oracle model to predict the functional activity of sampled sequence
    activity_sample_seqs = deepstarr(sample_seqs.permute(0,2,1))
    ## total activity level
    samples_total_activity = activity_sample_seqs.sum(dim=1)
    sorted_indices = torch.argsort(samples_total_activity, descending=True)
    ## subset the top 2,000 sequences
    top_sampled_seqs = sample_seqs[sorted_indices[:2000]]
    
    ## shap score for top activity sequences
    shap_score_top_sampled = gradient_shap(top_sampled_seqs, deepstarr)
    attribution_map_top_sampled = process_attribution_map(shap_score_top_sampled, k=6)
    mask_top_sampled = unit_mask(top_sampled_seqs)

    # entropic information for top sampled sequences
    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map_top_sampled], 
                                                                 top_sampled_seqs, 
                                                                 mask_top_sampled, 
                                                                 radius_count_cutoff=0.04)
    
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
    entropic_information_top_sampled = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
    
    # consistency across generated and observed sequence
    concatenated_seqs = torch.cat((X_test, sample_seqs), dim=0)
    shap_score_concatenated = gradient_shap(concatenated_seqs, deepstarr)
    attribution_map_concatenated = process_attribution_map(shap_score_concatenated, k=6)
    mask_concatenated = unit_mask(concatenated_seqs)

    phi_1_s, phi_2_s, r_s = spherical_coordinates_process_2_trad([attribution_map_concatenated], 
                                                                 concatenated_seqs, 
                                                                 mask_concatenated, 
                                                                 radius_count_cutoff=0.04)
    
    LIM, box_length, box_volume, n_bins, n_bins_half = initialize_integration_2(0.1)
    entropic_information_concatenated = calculate_entropy_2(phi_1_s, phi_2_s, r_s, n_bins, 0.1, box_volume, prior_range=3)
    
    # create a dictionary to store the results:
    results = {
        'entropic information of top 2000 activity sampled sequences': entropic_information_top_sampled,
        'entropic information of concatenated sequences': entropic_information_concatenated
    }
    
    filename = f'attribution_consistency_{current_date}.pkl'
    
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results have been saved to '{filename}'")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

