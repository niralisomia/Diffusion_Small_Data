import numpy as np
import pickle
from datetime import datetime

from PL_DeepSTARR import *
from seq_evals_improved import *
from helpers import *

if __name__=='__main__':
    samples_file_path = 'samples.npz'
    deepSTARR_data = 'DeepSTARR_data.h5'
    oracle_path = 'oracle_DeepSTARR_DeepSTARR_data.ckpt'

    deepstarr = load_deepstarr(oracle_path)

    # Get the current date and time
    current_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #load data
    x_test, x_synthetic, x_train = extract_data(samples_file_path, deepSTARR_data)

    x_test_tensor = numpy_to_tensor(x_test)
    x_synthetic_tensor = numpy_to_tensor(x_synthetic)
    x_train_tensor = numpy_to_tensor(x_train)

    x_test, x_synthetic = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)

    ##### Motif Enrichment #####

    x_synthetic_e = one_hot_to_seq(x_synthetic)
    x_test_e = one_hot_to_seq(x_test)

    create_fasta_file(x_synthetic_e,'sub_sythetic_seq.txt')
    create_fasta_file(x_test_e,'sub_test_seq.txt')

    motif_count_1 = motif_count('sub_test_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    motif_count_2 = motif_count('sub_sythetic_seq.txt', 'JASPAR2024_CORE_non-redundant_pfms_meme.txt')
    pr = enrich_pr(motif_count_1,motif_count_2)


    ##### Motif co-occurrence #####

    motif_matrix_1 = make_occurrence_matrix('sub_test_seq.txt')
    motif_matrix_2 = make_occurrence_matrix('sub_sythetic_seq.txt')

    mm_1 = np.array(motif_matrix_1).T
    mm_2 = np.array(motif_matrix_2).T

    C = np.cov(mm_1)
    C2 = np.cov(mm_2) 

    fn = frobenius_norm(C, C2)

    # Create a dictionary to store the results
    results = {
        'Pearson R Statistic': pr,
        'Frobenius Norm': fn
    }

    filename = f'motifs_{current_date}.pkl'

    # Save the results to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results have been saved to '{filename}'")
