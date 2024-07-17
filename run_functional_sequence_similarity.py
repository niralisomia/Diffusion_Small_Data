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

    #####Functional similarity: Conditional generation fidelity######

    #make predictions
    y_hat_test, y_hat_syn = load_predictions(x_test_tensor, x_synthetic_tensor, deepstarr)
    
    #calculate Conditional generation fidelity mse
    activity1 = y_hat_syn
    activity2 = y_hat_test
    mse = conditional_generation_fidelity(activity1, activity2)

    #####Functional similarity: Frechet distance#####

    embeddings1 = get_penultimate_embeddings(deepstarr, x_test_tensor)
    embeddings2 = get_penultimate_embeddings(deepstarr, x_synthetic_tensor)
    mu1, sigma1 = calculate_activation_statistics(embeddings1)
    mu2, sigma2 = calculate_activation_statistics(embeddings2)
    frechet_distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    #####Functional similarity: Predictive distribution shift#####

    hamming_distance = predictive_distribution_shift(x_synthetic_tensor, x_test_tensor)

    #####Sequence similarity: Percent identity#####

    x_synthetic_tensor2 = x_synthetic_tensor
    percent_identity_1 = calculate_cross_sequence_identity_batch(x_synthetic_tensor, x_synthetic_tensor2, batch_size=256)
    max_percent_identity_1 = np.max(percent_identity_1, axis=1)
    average_max_percent_identity_1 = np.mean(max_percent_identity_1)
    global_max_percent_identity_1 = np.max(max_percent_identity_1)

    percent_identity_2 = calculate_cross_sequence_identity_batch(x_synthetic_tensor, x_train_tensor, batch_size=2000)
    max_percent_identity_2 = np.max(percent_identity_2, axis=1)
    average_max_percent_identity_2 = np.mean(max_percent_identity_2) 
    global_max_percent_identity_2 = np.max(max_percent_identity_2)

    #####Sequence similarity: k-mer spectrum shift#####

    X_test, X_syn = put_deepstarr_into_NLA(x_test_tensor, x_synthetic_tensor)
    kmer_length=3
    Kullback_Leibler_divergence = kmer_statistics(kmer_length, X_test, X_syn)[0]
    Jensen_Shannon_distance = kmer_statistics(kmer_length, X_test, X_syn)[1]

    # Create a dictionary to store the results
    results = {
        'Conditional generation fidelity - mse': mse,
        'Frechet distance': frechet_distance,
        'Predictive distribution shift - Hamming_distance': hamming_distance,
        'Global max percent identity (samples v samples)': global_max_percent_identity_1,
        'Global max percent identity (samples v training)': global_max_percent_identity_2,
        'Average max percent identity (samples v samples)': average_max_percent_identity_1,
        'Average max percent identity (samples v training)': average_max_percent_identity_2,
        'kmer_spectra (Kullback Leibler divergence)': Kullback_Leibler_divergence, 
        'kmer_spectra (Jensen-Shannon distance)': Jensen_Shannon_distance
    }

    filename = f'functional_similarity_{current_date}.pkl'

    # Save the results to a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results have been saved to '{filename}'")