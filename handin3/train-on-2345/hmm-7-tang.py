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
    def __init__(self, init_probs=None, trans_probs=None, emission_probs=None):
        # Model parameters:
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs

        self.X_indices = []
        self.annotations = []
        self.Z_indices = []

    def add_data(self, X, ann):
        self.X_indices.append(X)
        self.annotations.append(ann)

    def save_hmm(self, fname):
        np.savez(fname, init_probs = self.init_probs, trans_probs = self.trans_probs, emission_probs = self.emission_probs)

    def load_hmm(self, fname):
        data = np.load(fname)
        self.init_probs = data['init_probs']#.flatten()
        self.trans_probs = data['trans_probs']
        self.emission_probs = data['emission_probs']

    def convert_ann_to_Z(self, ann):
        # ann -> [NNNNNNNNCCC......]
        N = len(ann)
        i = 0
        Z = []
        while i < N:
            if ann[i] == 'N':
                Z.append(3)
                i += 1
            elif ann[i] == 'C':
                Z += [2, 1, 0]
                i += 3
            else: # ann[i] == 'R'
                Z += [4, 5, 6]
                i += 3
        return Z


    def update_Z_indices(self, index='all'):
        if index == 'all':
            annotations = self.annotations
        else:
            annotations = self.annotations[index]  
        for ann in annotations:
            self.Z_indices.append(self.convert_ann_to_Z(ann))

    def count_transitions_and_emissions(self):
        """
        Returns a KxK matrix and a KxD matrix containing counts cf. above
        """
        K = 7 # 7 hidden states
        D = 4 # 4 emission states ACGT
        trasation_matrix = np.zeros((K, K))
        emission_matrix = np.zeros((K, D))

        count = 1
        for X, Z in zip(self.X_indices, self.Z_indices): # loop over training data
            print('counting : {}'.format(count))
            print(len(X), len(Z))
            for i, j in zip(Z[:-1], Z[1:]): # loop over hidden states
                i = int(i)
                j = int(j)
                trasation_matrix[i, j] += 1

            for index in range(len(X)): # loop over observation indices
                i = int(Z[index])
                k = int(X[index])
                emission_matrix[i, k] += 1
            count += 1
        return trasation_matrix, emission_matrix

    def training_by_counting(self):
        """
        Returns a HMM trained on x and z cf. training-by-counting.
        """
        trasation_matrix, emission_matrix = self.count_transitions_and_emissions()
        K, D = emission_matrix.shape

        init_probs = np.zeros(K)
        trans_probs = np.zeros((K, K))
        emission_probs = np.zeros((K, D))

        init_probs[3] = 1

        for i in range(trasation_matrix.shape[0]):
            for j in range(trasation_matrix.shape[1]):
                trans_probs[i, j] = trasation_matrix[i, j] / np.sum(trasation_matrix, axis=1)[i]

        for i in range(emission_matrix.shape[0]):
            for k in range(emission_matrix.shape[1]):
                emission_probs[i, k] = emission_matrix[i, k] / np.sum(emission_matrix, axis=1)[i]
        
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.emission_probs = emission_probs
        return hmm(init_probs, trans_probs, emission_probs)

    def compute_w_log(self, x):
        N = len(x) # number of observations
        K = 7
        w = np.zeros((K, N))
    
        #print(np.log(self.init_probs))
        #print(np.log(self.emission_probs[:, 0]))
        #print(np.log(self.init_probs) + np.log(self.emission_probs[:, 0]))
        #print(np.log(self.init_probs).shape)
        #print(np.log(self.emission_probs[:, 0]).shape)
        w[:, 0] = np.log(self.init_probs) + np.log(self.emission_probs[:, 0])
        for n in range(1, N):
            for k in range(K):
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

def test_train():
    hmm_7_states = hmm()

    fname_list = ['../genome{}.fa'.format(i) for i in [2, 3, 4, 5]]
    true_list = ['../true-ann{}.fa'.format(i) for i in [2, 3, 4, 5]]

    start = 0
    #end = 400000
    end = -1
    for fname, ture_name in zip(fname_list, true_list):
        genome = read_fasta_file(fname)
        keys = [key for key in genome.keys()]
        observations = genome[keys[0]]#[start:end]
        X = translate_observations_to_indices(observations)

        ann_true = read_fasta_file(ture_name)
        keys = [key for key in ann_true.keys()]
        ann = ann_true[keys[0]]#[start:end]

        hmm_7_states.add_data(X, ann)

    hmm_7_states.update_Z_indices()
    hmm_7_states.training_by_counting()
    hmm_7_states.save_hmm('hmm-7-counting.npz')
    #hmm_7_states.load_hmm('hmm-7-counting-1.npz')
    print(hmm_7_states.init_probs)
    print(hmm_7_states.trans_probs)
    print(hmm_7_states.emission_probs)

def test_predict():
    hmm_7_states = hmm()
    hmm_7_states.load_hmm('hmm-7-counting-1.npz')
    print(hmm_7_states.init_probs)
    print(hmm_7_states.trans_probs)
    print(hmm_7_states.emission_probs)
    fname_list = ['genome{}.fa'.format(i) for i in [1]]
    true_list = ['true-ann{}.fa'.format(i) for i in [1]]

    start = 0
    end = 1000
    for fname, ture_name in zip(fname_list, true_list):
        genome = read_fasta_file(fname)
        keys = [key for key in genome.keys()]
        observations = genome[keys[0]][start:end]
        X = translate_observations_to_indices(observations)
        ann_predicted = translate_indices_to_path(hmm_7_states.viterbi_decoding(X))

        ann_true = read_fasta_file(ture_name)
        keys = [key for key in ann_true.keys()]
        ann = ann_true[keys[0]][start:end]

        acc = compute_accuracy(ann, ann_predicted)
        print('{} accuracy: {}'.format(fname, acc))

def test_write_ann():
    hmm_7_states = hmm()
    hmm_7_states.load_hmm('hmm-7-counting.npz')
    print(hmm_7_states.init_probs)
    print(hmm_7_states.trans_probs)
    print(hmm_7_states.emission_probs)    

    fname_list = ['../genome{}.fa'.format(i) for i in [1]]
    pred_list = ['pred-ann{}.fa'.format(i) for i in [1]]

    start = 0
    end = -1
    for fname, pred_name in zip(fname_list, pred_list):
        print(fname)
        genome = read_fasta_file(fname)
        keys = [key for key in genome.keys()]
        observations = genome[keys[0]]#[start:end]
        X = translate_observations_to_indices(observations)
        ann_predicted = translate_indices_to_path(hmm_7_states.viterbi_decoding(X))

        with open(pred_name, 'w') as f:
            f.write('; some comment \n')
            f.write('>{} \n'.format(pred_name.split('.')[0]))

            length = 60
            for i in range(0, len(ann_predicted), length):
                f.write('{}\n'.format(ann_predicted[0+i:length+i]))
            f.close()

if __name__ == '__main__':
    #test_train()
    test_write_ann()


