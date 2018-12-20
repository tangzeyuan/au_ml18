import math  # Just ignore this :-)
import numpy as np
def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = np.array(init_probs)
        self.trans_probs = np.array(trans_probs)
        self.emission_probs = np.array(emission_probs)

init_probs_7_state = [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00]

trans_probs_7_state = [
    [0.00, 0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.05, 0.90, 0.05, 0.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00],
    [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00],
    [0.00, 0.00, 0.00, 0.10, 0.90, 0.00, 0.00],
]

emission_probs_7_state = [
    #   A     C     G     T
    [0.30, 0.25, 0.25, 0.20],
    [0.20, 0.35, 0.15, 0.30],
    [0.40, 0.15, 0.20, 0.25],
    [0.25, 0.25, 0.25, 0.25],
    [0.20, 0.40, 0.30, 0.10],
    [0.30, 0.20, 0.30, 0.20],
    [0.15, 0.30, 0.20, 0.35],
]

hmm_7_state = hmm(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))

def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def count_transitions_and_emissions(K, D, x, z):
    """
    Returns a KxK matrix and a KxD matrix containing counts cf. above
    """
    trasation_matrix = np.zeros((K, K))
    emission_matrix = np.zeros((K, D))
    mapping = ['a', 'c', 'g', 't']
    x_indices = [mapping.index(item.lower()) for item in x]

    for i, j in zip(z[:-1], z[1:]):
        i = int(i)
        j = int(j)
        trasation_matrix[i][j] += 1

    for index in range(len(x)):
        i = int(z[index])
        k = x_indices[index]
        emission_matrix[i][k] += 1
    
    return trasation_matrix, emission_matrix

x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'

x_test = x_long[:10]
z_test = z_long[:10]
trasation_matrix, emission_matrix = count_transitions_and_emissions(7, 4, x_test, z_test)
print(trasation_matrix)
print(emission_matrix)

def training_by_counting(K, D, x, z):
    """
    Returns a HMM trained on x and z cf. training-by-counting.
    """
    trasation_matrix, emission_matrix = count_transitions_and_emissions(K, D, x, z)

    init_probs = np.zeros((K, 1))
    trans_probs = np.zeros((K, K))
    emission_probs = np.zeros((K, D))

    start_index = int(z[0])
    init_probs[start_index] = 1

    for i in range(trasation_matrix.shape[0]):
        for j in range(trasation_matrix.shape[1]):
            trans_probs[i][j] = trasation_matrix[i][j] / np.sum(trasation_matrix, axis=1)[i]

    for i in range(emission_matrix.shape[0]):
        for k in range(emission_matrix.shape[1]):
            emission_probs[i][k] = emission_matrix[i][k] / np.sum(emission_matrix, axis=1)[i]
    
    return hmm(init_probs, trans_probs, emission_probs)

def compute_w_log(model, x):
    K = len(model.init_probs)
    N = len(x)
    
    # w[k][n] = w(zn), k x n
    # w = make_table(K, N)
    w = np.zeros((K, N))
    
    # Base case: fill out w[i][0] for i = 0..k-1
    for i in range(K):
        w[i, 0] = log(model.init_probs[i]) + log(model.emission_probs[i, x[0]])
    
    # Inductive case: fill out w[i][j] for i = 0..k, j = 1..n
    for n in range(1, N):
        for k in range(K):
            w[k, n] = -np.inf  # log probs go negative
            for j in range(K):         
                w[k, n] = max(w[k, n], log(model.emission_probs[k, x[n]]) + w[j, n-1] + log(model.trans_probs[j, k]))

    return w

def opt_path_prob_log(w):
    return max([row[-1] for row in w])

def backtrack_log(w, model, x):
    N = w.shape[1]
    # z[1..N] = undef
    z = np.empty((N), dtype=int)
    
    # z[N] = arg max_k w[k][N]
    z[N-1] = np.argmax(w[:, N-1])

    # z[n] = arg max_k (p(x[n+1]|z[n+1]) * w[k][n] * p(z[n+1]|k))
    for n in range(N-2, -1, -1):
        emission_prob = log(model.emission_probs[z[n+1], x[n+1]])
        trans_probs = [log(row[z[n+1]]) for row in model.trans_probs]
        z[n] = np.argmax(emission_prob + w[:, n] + trans_probs)

    return z

hmm_7_state_tbc = training_by_counting(7, 4, x_long, z_long)
print(hmm_7_state_tbc.init_probs)
print(hmm_7_state_tbc.trans_probs)
print(hmm_7_state_tbc.emission_probs)


print(x_long[:10])
x_long = translate_observations_to_indices(x_long)
print(x_long[:10])

w = compute_w_log(hmm_7_state, x_long)
z_vit = backtrack_log(w, hmm_7_state, x_long)

w_tbc = compute_w_log(hmm_7_state_tbc, x_long)
z_vit_tbc = backtrack_log(w_tbc, hmm_7_state_tbc, x_long)

# Your comparison of z_vit and z_vit_tbc here ...
print(z_vit)
print(z_vit_tbc)