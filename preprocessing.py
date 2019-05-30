import csv
import scipy.io as sio
import numpy as np
import matlab
from sklearn.utils import shuffle
from collections import Counter

# =============================================================================
# 
# 1-1. Load Data
# 
# =============================================================================

def load_data(data_path):
    data = []
    labels = []
    with open(data_path + '/REFERENCE.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            file_name = row[0]
            label = row[1]

            test = sio.loadmat(data_path + '/' + file_name + '.mat')
            content = test['val'][0]

            data.append(content)
            labels.append(label)    

    return data, labels

# =============================================================================
# 
# 1-2. Categorizing
# 
# =============================================================================

def format_labels(labels):

    __mapping__ = {
        'A': 2, # AF rhythm
        'O': 2, # Other rhythm
        '~': 2, # Noise rhythm
        'N': 0 # Normal rhythm
    }

    return [__mapping__[x] for x in labels]

# =============================================================================
# 
# 1-3. Balancing
# 
# =============================================================================

def balance(x, y):
    uniq = np.unique(y)
    selected = dict()

    print('Before balancing it : {}\n'.format(Counter(y)))

    for val in uniq:
        selected[val] = [x[i] for i in matlab.find(y==val)]
#    min_len = 6 * min([len(x) for x in selected.values()]) 최대 6배까지 허용할 경우
    min_len = 1 * min([len(x) for x in selected.values()]) # 동일한 수로 조정
    x = []
    y = []

    for (key, value) in selected.items():
        slen = min(len(value), min_len)
        x += value[:slen]
        y += [key for i in range(slen)]

    x, y = shuffle(x, y)
    print('After balancing it  : {}'.format(Counter(y)))

    return x, y

# =============================================================================
# 
# 1-4. Normalizing
# 
# =============================================================================

def normalize_ecg(ecg):
    """
    Normalizes to a range of [-1; 1]
    Param ecg: input signal
    Return: normalized signal
    """

    ecg = ecg-np.mean(ecg)
    ecg = ecg / max(np.fabs(np.min(ecg)), np.fabs(np.max(ecg)))
    
    return ecg