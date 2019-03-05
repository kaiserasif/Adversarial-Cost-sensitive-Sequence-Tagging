from __future__ import print_function

import subprocess
import sys, os
from concurrent.futures import ProcessPoolExecutor

from sklearn.model_selection import KFold
import itertools

import numpy as np


svm_learn = './svm_multiclass_learn'
svm_classify = './svm_multiclass_classify'
cost_file = None

def save_lines(filename, sequences):
    with open(filename, 'wt') as f:
        for lines in sequences:
            for line in lines:
                f.write(line)
            f.write('\n')


def find_diff_of_max_2(nparray):
    nparray[::-1].sort()
    return nparray[0] - nparray[1]


def train_svm(C, datafile, modelfile, print_stdout=False):
    options = ['-v', '1', '-c', str(C)]
    if cost_file: options += ['--cost_file', cost_file]
    svm_learn_param = [svm_learn] + options + [datafile, modelfile ]
    # subprocess.call(svm_learn_param)
    # print(' '.join(svm_learn_param))
    cmd = subprocess.Popen(svm_learn_param, stdout=subprocess.PIPE)
    stdout, stderr = cmd.communicate()
    if print_stdout:
        if stdout: print (stdout.decode('utf-8'))
        if stderr: print (stderr.decode('utf-8'))
    

def test_svm(datafile, modelfile, outputfile, print_stdout=False):
    options = []
    if cost_file: options = ['--cost_file', cost_file]
    svm_classify_param = [svm_classify] + options + [datafile, modelfile, outputfile]
    # print(' '.join(svm_classify_param))
    # subprocess.call(svm_classify_param)
    cmd = subprocess.Popen(svm_classify_param, stdout=subprocess.PIPE)
    stdout, stderr = cmd.communicate()
    if print_stdout:
        if stdout: print (stdout.decode('utf-8'))
        if stderr: print (stderr.decode('utf-8'))
    line = stdout.decode('utf-8').split('\n')[-2].strip()
    loss = float(line.split()[-1])
    # print line, loss
    return loss


def parallel_executor(C, data_tr, data_ts):
    tmpmodel = '%f%s%s.model' % (C, data_tr, data_ts)
    tmpout = '%f%s%s.out' % (C, data_tr, data_ts)
    train_svm(C, data_tr, tmpmodel)
    test_err = test_svm(data_ts, tmpmodel, tmpout)
    try:
        os.remove(tmpmodel)
        os.remove(tmpout)
    except:
        pass
    return test_err


def cross_validate(sequences, kf, C):
    tmpdata_tr, tmpdata_ts = 'cross_validate_tmpdata_tr.txt', 'cross_validate_tmpdata_ts.txt'
    kfold = KFold(kf, shuffle=False) # don't shuffle keep consistent splits across algorithms
    errors = np.zeros(len(C))
    # for each fold, we create train test files to run in svm
    # so minimize this 
    # for each fold, iterate through the (C, gamma)
    for train_index, test_index in kfold.split(sequences):
        data_tr = [sequences[i] for i in train_index]
        data_ts = [sequences[i] for i in test_index]
        # save splits to file
        save_lines(tmpdata_tr, data_tr)
        save_lines(tmpdata_ts, data_ts)
        
        with ProcessPoolExecutor(max_workers=8) as pool:
            n = len(C)
            for i, test_err in enumerate(pool.map(parallel_executor, C, 
                [tmpdata_tr]*n, [tmpdata_ts]*n) ):
                errors[i] += test_err
    try:
        os.remove(tmpdata_tr)
        os.remove(tmpdata_ts)
    except:
        pass

    return C[errors.argmin()]
    

def find_best_C(sequences):

    kf = 5
    # # two stage search
    # Cs = [2.0**i for i in range(-8, 8+1, 4)]
    # C0 = cross_validate(X, y, kf, Cs )
    # print ('best C0: {}'.format(C0))
    Cs =  [10.**p for p in range(-4, 4+1) ]
    C = cross_validate(sequences, kf, Cs) 
    print ('best C: {}'.format(C))

    return C


def run_experiment(data_dir):

    # load the sequence indices to be used for cross validation
    # for consistency across models
    validation_indices = np.loadtxt(os.path.join(data_dir, 'validation_set.txt'), delimiter=',').astype(int)

    # load data as sequence of lines .. each line containing <y> 1:feature1 2:feature2.. etc.
    # we need only to shuffle the sequences, don't need to parse the values
    # [[list of lines for sequence 1], [list of sequence 2 lines], ...]
    train_file = os.path.join(data_dir, 'train.dat')
    test_file = os.path.join(data_dir, 'test.dat')
    train_sequences = []
    with open(train_file, 'rt') as f:
        seq = []
        for line in f.readlines():
            if len(line.strip()):
                seq.append(line)
            else:
                if seq: train_sequences.append(seq)
                seq = []
        if seq: train_sequences.append(seq)
    
    C = find_best_C( [train_sequences[i] for i in validation_indices] )

    # run main train test
    train_svm(C, train_file, 'model', print_stdout=True) # train
    test_err = test_svm(test_file, 'model', 'prediction.txt', print_stdout=True) # test
    print ('test loss:', test_err)
    sys.stdout.flush()


def run():
    global svm_learn, svm_classify, cost_file
    svm_learn, svm_classify, data_dir = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), os.path.abspath(sys.argv[3])
    if len(sys.argv) > 4: cost_file = sys.argv[4]
    run_experiment(data_dir)


if __name__ == '__main__':
    run()

