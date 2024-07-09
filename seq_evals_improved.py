import numpy as np
import scipy
from scipy import linalg

#############################################################################################
# Functional similarity: Conditional generation fidelity
#############################################################################################
'''
prerequisite: 
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A)) 
    - oracle (inference model that maps x to activities)

example:
    activity1 = oracle.predict(x_synthetic)
    activity2 = oracle.predict(x_test)
    mse = conditional_generation_fidelity(activity1, activity2)
'''

def conditional_generation_fidelity(activity1, activity2):
    return np.mean((activity1 - activity2)**2)

#############################################################################################
# Functional similarity: Frechet distance
#############################################################################################
'''
prerequisite: 
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A)) 
    - oracle_embedding_fun (function that acquires the penultimate embeddings)

example:
    embeddings1 = oracle_embedding_fun(x_synthetic)
    embeddings2 = oracle_embedding_fun(x_test)
    mu1, sigma1 = calculate_activation_statistics(embeddings1)
    mu2, sigma2 = calculate_activation_statistics(embeddings2)
    distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
'''

def calculate_activation_statistics(embeddings):
    embeddings_d = embeddings.detach().numpy()
    mu = np.mean(embeddings_d, axis=0)
    sigma = np.cov(embeddings_d, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    #Frechet distance: d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

#############################################################################################
# Functional similarity: Predictive distribution shift
#############################################################################################
'''
prerequisite:
    - x_synthetic_tensor (generated seqeunces as a tensor with shapes (N,L,A)) 
    - x_test_tensor (observed sequences as a tensor with shapes (N,L,A))

example:
    activity1 = oracle.predict(x_synthetic)
    activity2 = oracle.predict(x_test)
    mse = conditional_generation_fidelity(activity1, activity2)
'''

def predictive_distribution_shift(x_synthetic_tensor, x_test_tensor):
    
    #encode bases using 0,1,2,3 (eliminate a dimension)
    base_indices_test = np.argmax(x_test_tensor.detach().numpy(), axis=1)
    base_indices_syn = np.argmax(x_synthetic_tensor.detach().numpy(), axis=1)

    #flatten the arrays (now they are one dimension)
    base_indices_test_f = base_indices_test.flatten()
    base_indices_syn_f = base_indices_syn.flatten()

    #return ks test statistic
    return scipy.stats.ks_2samp(base_indices_syn_f, base_indices_test_f).statistic