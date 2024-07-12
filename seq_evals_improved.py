import numpy as np
from numpy import load
import h5py
import pandas as pd
import torch
import lightning as L
from pytorch_lightning import LightningModule
from scipy import linalg
import scipy
from tqdm import tqdm
from itertools import product
import tensorflow as tf
from tensorflow import keras
from six.moves import cPickle
from sklearn.metrics import mean_squared_error
import gc
from Bio import motifs
import os
from pymemesuite import fimo
from pymemesuite.common import MotifFile, Sequence
from pymemesuite import fimo
from pymemesuite.fimo import FIMO
from Bio import SeqIO
import glob
import tempfile
import Bio




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

#############################################################################################
# Sequence similarity: Percent identity
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))

example:
    percent_identity = calculate_cross_sequence_identity_batch(x_synthetic, x_test, batch_size)
    max_percent_identity = np.max(percent_identity, axis=1)
    global_max_percent_identity = np.max(max_percent_identity)
'''

def calculate_cross_sequence_identity_batch(X_train, X_test, batch_size):
    num_train, seq_length, alphabet_size = X_train.shape    
    num_test = X_test.shape[0]
    
    # Reshape the matrices for dot product computation
    X_train = np.reshape(X_train, [-1, seq_length * alphabet_size])
    X_test = np.reshape(X_test, [-1, seq_length * alphabet_size])
    
    # Initialize the matrix to store the results
    seq_identity = np.zeros((num_train, num_test)).astype(np.int8)
    
    # Process the training data in batches
    for start_idx in tqdm(range(0, num_train, batch_size)):
        end_idx = min(start_idx + batch_size, num_train)
        
        # Compute the dot product for this batch
        batch_result = np.dot(X_train[start_idx:end_idx], X_test.T) 
        
        # Store the result in the corresponding slice of the output matrix
        seq_identity[start_idx:end_idx, :] = batch_result.astype(np.int8)
    
    return seq_identity

#############################################################################################
# Sequence similarity: k-mer spectrum shift
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))

example:
    kld, jsd = kmer_statistics(kmer_len, data1, data2)
'''

def kmer_statistics(kmer_length, data1, data2):

    #generate kmer distributions 
    dist1 = compute_kmer_spectra(data1, kmer_length)
    dist2 = compute_kmer_spectra(data2, kmer_length)

    #computer KLD
    kld = np.round(np.sum(scipy.special.kl_div(dist1, dist2)), 6)

    #computer jensen-shannon 
    jsd = np.round(np.sum(scipy.spatial.distance.jensenshannon(dist1, dist2)), 6)

    return kld, jsd

def compute_kmer_spectra(
    X,
    kmer_length=3,
    dna_dict = {
        0: "A",
        1: "C",
        2: "G",
        3: "T"
      }
    ):
    # convert one hot to A,C,G,T
    seq_list = []

    for index in tqdm(range(len(X))): #for loop is what actually converts a list of one-hot encoded sequences into ACGT

        seq = X[index]

        seq_list += ["".join([dna_dict[np.where(i)[0][0]] for i in seq])]

    obj = kmer_featurization(kmer_length)  # initialize a kmer_featurization object
    kmer_features = obj.obtain_kmer_feature_for_a_list_of_sequences(seq_list, write_number_of_occurrences=True)

    kmer_permutations = ["".join(p) for p in product(["A", "C", "G", "T"], repeat=kmer_length)] #list of all kmer permutations, length specified by repeat=

    kmer_dict = {}
    for kmer in kmer_permutations:
        n = obj.kmer_numbering_for_one_kmer(kmer)
        kmer_dict[n] = kmer

    global_counts = np.sum(np.array(kmer_features), axis=0)

    # what to compute entropy against
    global_counts_normalized = global_counts / sum(global_counts) # this is the distribution of kmers in the testset
    # print(global_counts_normalized)
    return global_counts_normalized

class kmer_featurization:

    def __init__(self, k):
        """
        seqs: a list of DNA sequences
        k: the "k" in k-mer
        """
        self.k = k
        self.letters = ['A', 'C', 'G', 'T']
        self.multiplyBy = 4 ** np.arange(k-1, -1, -1) # the multiplying number for each digit position in the k-number system
        self.n = 4**k # number of possible k-mers

    def obtain_kmer_feature_for_a_list_of_sequences(self, seqs, write_number_of_occurrences=False):
        """
        Given a list of m DNA sequences, return a 2-d array with shape (m, 4**k) for the 1-hot representation of the kmer features.
        Args:
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        kmer_features = [] #a list containing the one-hot representation of kmers for each sequence in the list of sequences given
        for seq in seqs: #first obtain the one-hot representation of the kmers in a sequence
            this_kmer_feature = self.obtain_kmer_feature_for_one_sequence(seq.upper(), write_number_of_occurrences=write_number_of_occurrences)
            kmer_features.append(this_kmer_feature) #append this one-hot list into another list

        kmer_features = np.array(kmer_features)

        return kmer_features

    def obtain_kmer_feature_for_one_sequence(self, seq, write_number_of_occurrences=False): #
        """
        Given a DNA sequence, return the 1-hot representation of its kmer feature.
        Args:
          seq:
            a string, a DNA sequence
          write_number_of_occurrences:
            a boolean. If False, then in the 1-hot representation, the percentage of the occurrence of a kmer will be recorded; otherwise the number of occurrences will be recorded. Default False.
        """
        number_of_kmers = len(seq) - self.k + 1

        kmer_feature = np.zeros(self.n) #array of zeroes the same length of all possible kmers

        for i in range(number_of_kmers): #for each kmer feature, turn the corresponding index in the list of all kmer features to 1
            this_kmer = seq[i:(i+self.k)]
            this_numbering = self.kmer_numbering_for_one_kmer(this_kmer)
            kmer_feature[this_numbering] += 1

        if not write_number_of_occurrences:
            kmer_feature = kmer_feature / number_of_kmers

        return kmer_feature

    def kmer_numbering_for_one_kmer(self, kmer): #returns the corresponding index of a kmer in the larger list of all possible kmers?
        """
        Given a k-mer, return its numbering (the 0-based position in 1-hot representation)
        """
        digits = []
        for letter in kmer:
            digits.append(self.letters.index(letter))

        digits = np.array(digits)

        numbering = (digits * self.multiplyBy).sum()

        return numbering

#############################################################################################
# Sequence similarity: Discriminatability
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_train (observed training sequences with shapes (N,L,A))
    - classification model 

example:
    data_dict = prep_data_for_classification(x_train, x_synthetic)
    write_to_h5('Discriminatability.h5', data_dict)

    run train_deepstarr_ood_score.py on elzar
'''

def prep_data_for_classification(x_test_tensor, x_synthetic_tensor):
    x_train = np.vstack([x_test_tensor.detach().numpy(), x_synthetic_tensor.detach().numpy()])
    y_train = np.vstack([np.ones((x_test_tensor.shape[0],1)), np.zeros((x_synthetic_tensor.shape[0],1))])
    x_train = np.transpose(x_train, (0, 2, 1)) 

    #write x_train and y_train into dict to create .h5 file
    data_dict = {
        'x_train': x_train,
        'y_train': y_train,
    }

    return data_dict

#############################################################################################
# Compositional similarity: Motif enrichment
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))
    - JASPAR_file (Datasbase for motif search)

example:
	x_synthetic = one_hot_to_seq(x_synthetic)
	x_test = one_hot_to_seq(x_test)
	create_fasta_file(x_synthetic,'sythetic_seq.txt')
	create_fasta_file(x_test,'test_seq.txt')
	motif_count = motif_count('test_seq.txt', JASPAR_file)
	motif_count_2 = motif_count('synthetic_seq.txt', JASPAR_file)
	pr = enrich_pr(motif_count,motif_count_2)
'''

def motif_count(path, path_to_database):
    '''
    path is the filepath to the list of sequences in fasta format

    returns a dictionary containing the motif counts for all the sequences
    '''

    motif_ids = []
    occurrence = []

    sequences = [
        Sequence(str(record.seq), name=record.id.encode())
        for record in Bio.SeqIO.parse(path, "fasta")
        ]
    
    fimo = FIMO() 
    with MotifFile("JASPAR2024_CORE_non-redundant_pfms_meme.txt") as motif_file:
        for motif in motif_file: 
            pattern = fimo.score_motif(motif, sequences, motif_file.background)
            motif_ids.append(motif.accession.decode())
            occurrence.append(len(pattern.matched_elements))
    
    motif_counts = dict(zip(motif_ids,occurrence))

    return motif_counts

def enrich_pr(count_1,count_2):
	c_1 = list(count_1.values())
	c_2 = list(count_2.values())

	return scipy.stats.pearsonr(c_1,c_2)

#############################################################################################
# Compositional similarity: Motif co-occurrence
#############################################################################################
'''
prerequisite:
    - x_synthetic (generated seqeunces with shapes (N,L,A)) 
    - x_test (observed sequences with shapes (N,L,A))
    - JASPAR_file (Datasbase for motif search)

example:

	motif_matrix_test = make_occurrence_matrix('test_seq.txt')
    motif_matrix_synthetic = make_occurrence_matrix('synthetic_seq.txt')
    C = np.cov(motif_matrix_test)
    C2 = np.cov(motif_matrix_synthetic)
    distance = frobenius_norm(C, C2)
'''

def make_occurrence_matrix(path):
    '''
    path is the filepath to the list of sequences in fasta format

    returns a matrix containing the motif counts for all the sequences
    '''

    motif_ids = []
    occurrence = []

    sequences = [
        Sequence(str(record.seq), name=record.id.encode())
        for record in Bio.SeqIO.parse(path, "fasta")
        ]

    
    fimo = FIMO() 
    #matrix with m rows and n columns
    occurrence_matrix = []
    for sequence in sequences:
        sequence = [sequence]
        occurrence = []
        motif_ids = []
        with MotifFile("JASPAR2024_CORE_non-redundant_pfms_meme.txt") as motif_file:
            for motif in motif_file: 
                pattern = fimo.score_motif(motif, sequence, motif_file.background)
                motif_ids.append(motif.accession.decode())
                occurrence.append(len(pattern.matched_elements))
        occurrence_matrix.append(occurrence)

    return occurrence_matrix


def covariance_matrix(x):
    return np.cov(x)


def frobenius_norm(cov, cov2):
    return np.sqrt(np.sum((cov - cov2)**2))