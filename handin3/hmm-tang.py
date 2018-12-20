import numpy as np

def read_fasta_file(filename):
    """
    Reads the given FASTA file f and returns a dictionary of sequences.

    Lines starting with ';' in the FASTA file are ignored.
    """
    sequences_lines = {}
    current_sequence_lines = None
    with open(filename) as fp:
        for line in fp:
            line = line.strip()
            if line.startswith(';') or not line:
                continue
            if line.startswith('>'):
                sequence_name = line.lstrip('>')
                current_sequence_lines = []
                sequences_lines[sequence_name] = current_sequence_lines
            else:
                if current_sequence_lines is not None:
                    current_sequence_lines.append(line)
    sequences = {}
    for name, lines in sequences_lines.items():
        sequences[name] = ''.join(lines)
    return sequences

def translate_indices_to_path(indices):
    mapping = ['C', 'C', 'C', 'N', 'R', 'R', 'R']
    return ''.join([mapping[i] for i in indices])

def translate_observations_to_indices(obs):
    mapping = {'a': 0, 'c': 1, 'g': 2, 't': 3}
    return [mapping[symbol.lower()] for symbol in obs]

def translate_indices_to_observations(indices):
    mapping = ['a', 'c', 'g', 't']
    return ''.join(mapping[idx] for idx in indices)

def compute_accuracy(true_ann, pred_ann):
    if len(true_ann) != len(pred_ann):
        return 0.0
    return sum(1 if true_ann[i] == pred_ann[i] else 0 
               for i in range(len(true_ann))) / len(true_ann)

class hmm:
    def __init__(self, init_probs, trans_probs, emission_probs):

        # Model parameters:
        self.init_probs = np.array(init_probs)
        self.trans_probs = np.array(trans_probs)
        self.emission_probs = np.array(emission_probs)

        # Sizes:
        self.K = len(self.init_probs)
        self.D = len(self.emission_probs[1])

    def compute_w_log(self, x):
        N = len(x)
    
        w = np.zeros((self.K, N))
    
        w[:, 0] = np.log(self.init_probs) + np.log(self.emission_probs[:, 0])
        for n in range(1, N):
            for k in range(self.K):
                w[k, n] = np.log(self.emission_probs[k, x[n]])+np.max(w[:, n-1]
                        +np.log(self.trans_probs[:, k]))
        self.w = w

    def opt_path_prob_log(self, w):
        p = np.max(w[:, -1])
    
    def backtrack_log(self, x):
        N = self.w.shape[1]
        Z = np.zeros(N, dtype=int)
        Z[-1] = np.argmax(self.w[:, -1])
        for n in range(N-2, -1, -1):
            Z[n] = np.argmax(np.log(self.emission_probs[Z[n+1],x[n+1]])+self.w[:, n]
                             +np.log(self.trans_probs[:, Z[n+1]]))
        return Z

    def viterbi_decoding(self, x):
        self.compute_w_log(x)
        return self.backtrack_log(x)

    
def counting(X_obs, Z_true):
    # X_obs -> [TTGTTGATATTC......]
    # Z_true -> [NNNNNNNNNNNN......]
    n_Z = len(Z_true)
    range_Z = np.arange(n_Z)
    count_NN = 0
    for a, b in zip(Z_true[:-1], Z_true[1:]):
        ''' 
        0 1
        1 2
        2 3
        3 4
        '''
        if a == 'N' and b =='N':
            count_NN += 1

    count_N_CCC_ATG = 0
    count_N_CCC_GTG = 0
    count_N_CCC_TTG = 0
    count_N_RRR_TTA = 0
    count_N_RRR_CTA = 0
    count_N_RRR_TCA = 0

    for a, b, c, d in zip(range_Z[:-4], range_Z[1:-3], range_Z[2:-2], range_Z[3:-1]):
        '''
        0 1 2 3
        1 2 3 4
        2 3 4 5
        3 4 5 6
        '''
        quadruple = ''.join([Z_true[i] for i in [a, b, c, d]])
        if quadruple == 'NCCC':
            sub_obs = ''.join([X_obs[i] for i in [b, c, d]])
            if sub_obs == 'ATG':
                count_N_CCC_ATG += 1
            if sub_obs == 'GTG':
                count_N_CCC_GTG += 1
            if sub_obs == 'TTG':
                count_N_CCC_TTG += 1
        
        if quadruple == 'NRRR':
            sub_obs = ''.join([X_obs[i] for i in [b, c, d]])
            if sub_obs == 'TTA':
                count_N_RRR_TTA += 1
            if sub_obs == 'CTA':
                count_N_RRR_CTA += 1
            if sub_obs == 'TCA':
                count_N_RRR_TCA += 1       

    return (count_NN, count_N_CCC_ATG, count_N_CCC_GTG, count_N_CCC_TTG)
    

if __name__ == '__main__':

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
    
    hmm7 = hmm(init_probs_7_state, trans_probs_7_state, emission_probs_7_state)

    start = 0
    end = 300

    

    names = ['genome1.fa', 'genome2.fa']
    keys = ['genome1', 'genome2']
    true_names = ['true-ann1.fa', 'true-ann2.fa']
    
    for name, tru in zip(names, true_names):
        g = read_fasta_file(name)
        keys = [key for key in g.keys()]
        g = g[keys[0]][start:end]

        x_indices = translate_observations_to_indices(g)
        b = hmm7.viterbi_decoding(x_indices)
        c = translate_indices_to_path(b)
        Z = translate_indices_to_path(hmm7.viterbi_decoding(translate_observations_to_indices(g)))

        print(g)
        print(x_indices)
        print(b)
        print(c)
        print(Z)
        Ztru = read_fasta_file(tru)
        keys = [key for key in Ztru.keys()]
        Ztru = Ztru[keys[0]][start:end]
        

        a = compute_accuracy(Ztru, Z)
        print(Ztru)
        print(a)

        print(counting(g, Ztru))
        print('\n')
        
        
        
        
    

