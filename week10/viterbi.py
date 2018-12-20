import numpy as np

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):
        self.init_probs = np.array(init_probs)
        self.trans_probs = np.atleast_2d(trans_probs)
        self.emission_probs = np.atleast_2d(emission_probs)

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

def translate_path_to_indices(path):
    return list(map(lambda x: int(x), path))

def translate_indices_to_path(indices):
    return ''.join([str(i) for i in indices])

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

# Collect the matrices in a class.
hmm_7_state = hmm(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)


def compute_w(model, x):
    K = len(model.init_probs) # number of states
    N = len(x) # number of observations
    
    # w[k][n] = w(zn), k x n
    # w = make_table(K, N)
    w = np.zeros((K, N))
    
    # Base case: fill out w[i][0] for i = 0..k-1
    # w(z1) = p(z1)p(x1|z1)
    for i in range(K):
        w[i, 0] = model.init_probs[i]*model.emission_probs[i, x[0]]
    
    
    # Inductive case: fill out w[i][j] for i = 0..k, j = 1..n
    # w(zn) = p(xn|zn)*max_(zn-1) w(zn-1)p(zn|zn-1)
    for n in range(1, N):
        for k in range(K):
            for j in range(K):
                w[k, n] = max(w[k, n], model.emission_probs[k, x[n]] * w[j, n-1] * model.trans_probs[j, k])

    return w

def compute_w_tang(model, x):
    K = len(model.init_probs)
    N = len(x)
    
    w = np.zeros((K, N))
    
    # Base case: fill out w[i][0] for i = 0..k-1
    # ...
    for i in range(K):
        w[i, 0] = model.init_probs[i] * model.emission_probs[i, 0]
    
    # Inductive case: fill out w[i][j] for i = 0..k, j = 0..n-1
    # ...
    for k in range(K):
        for n in range(1, N):
            w[k, n] = max(w[k, n], w[k-1, n-1] * model.emission_probs[k, x[n]] * model.trans_probs[k-1, k])
    return w

def compute_w_log(model, x):
    K = len(model.init_probs)
    N = len(x)

    w = np.zeros((K, N))

    w[:, 0] = np.log(model.init_probs) + np.log(model.emission_probs[:, 0])
    for n in range(1, N):
        for k in range(K):
            w[k, n] = np.log(model.emission_probs[k, x[n]])+np.max(w[:, n-1]
                    +np.log(model.trans_probs[:, k]))
    return w

def opt_path_prob(w):
    return np.max(w[:, -1])

x_short = 'GTTTCCCAGTGTATATCGAGGGATACTACGTGCATAGTAACATCGGCCAA'
z_short = '33333333333321021021021021021021021021021021021021'

x_long = 'TGAGTATCACTTAGGTCTATGTCTAGTCGTCTTTCGTAATGTTTGGTCTTGTCACCAGTTATCCTATGGCGCTCCGAGTCTGGTTCTCGAAATAAGCATCCCCGCCCAAGTCATGCACCCGTTTGTGTTCTTCGCCGACTTGAGCGACTTAATGAGGATGCCACTCGTCACCATCTTGAACATGCCACCAACGAGGTTGCCGCCGTCCATTATAACTACAACCTAGACAATTTTCGCTTTAGGTCCATTCACTAGGCCGAAATCCGCTGGAGTAAGCACAAAGCTCGTATAGGCAAAACCGACTCCATGAGTCTGCCTCCCGACCATTCCCATCAAAATACGCTATCAATACTAAAAAAATGACGGTTCAGCCTCACCCGGATGCTCGAGACAGCACACGGACATGATAGCGAACGTGACCAGTGTAGTGGCCCAGGGGAACCGCCGCGCCATTTTGTTCATGGCCCCGCTGCCGAATATTTCGATCCCAGCTAGAGTAATGACCTGTAGCTTAAACCCACTTTTGGCCCAAACTAGAGCAACAATCGGAATGGCTGAAGTGAATGCCGGCATGCCCTCAGCTCTAAGCGCCTCGATCGCAGTAATGACCGTCTTAACATTAGCTCTCAACGCTATGCAGTGGCTTTGGTGTCGCTTACTACCAGTTCCGAACGTCTCGGGGGTCTTGATGCAGCGCACCACGATGCCAAGCCACGCTGAATCGGGCAGCCAGCAGGATCGTTACAGTCGAGCCCACGGCAATGCGAGCCGTCACGTTGCCGAATATGCACTGCGGGACTACGGACGCAGGGCCGCCAACCATCTGGTTGACGATAGCCAAACACGGTCCAGAGGTGCCCCATCTCGGTTATTTGGATCGTAATTTTTGTGAAGAACACTGCAAACGCAAGTGGCTTTCCAGACTTTACGACTATGTGCCATCATTTAAGGCTACGACCCGGCTTTTAAGACCCCCACCACTAAATAGAGGTACATCTGA'
z_long = '3333321021021021021021021021021021021021021021021021021021021021021021033333333334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210210321021021021021021021021033334564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564564563333333456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456456332102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102102103210210210210210210210210210210210210210210210210210210210210210'


w = compute_w_tang(hmm_7_state, translate_observations_to_indices(x_short[:10]))
log_w = compute_w_log(hmm_7_state, translate_observations_to_indices(x_short[:10]))
print(np.log(w))
print(log_w)
print(w.shape)
#print(opt_path_prob(w))