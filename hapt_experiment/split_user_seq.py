"""
Data set: http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions
Data contains 61 sequences (experiements) as dictated by column 1 of Rawdata.txt
1) 61 in train-test split doesn't seem sufficient
2) From Rawdata to extracted data (train and test provided) was needed
   perhaps could be done by length ratio, but still gives very long and 
   few samples
3) Too long sequences proved intractable 

File read : subject_id_train.txt or subject_id_test.txt

We first, separate sequences based on subject_id
Then random length choosen between provided ranges
And saved into a file name provided
outputs: subject_id, start_index, end_index (0 based and inclusive)
"""

import sys
import numpy as np

Usage = 'python <thisprogram> <useridfile> <outputfile> <minlen> <maxlen>'

if len(sys.argv) != 5:
    print (Usage)
    exit(1)

def split(seq, low=100, high=200):
    print( len(seq) )
    if len(seq) < high: return [seq]
    ret = []
    while len(seq) > 0:
        l = np.random.randint(low, high)
        if (len(seq) - l < low): l = high # remaining seq too short
        ret.append(seq[:l])
        seq = seq[l:]
    return ret

try:
    infile, outfile = sys.argv[1:2+1]
    minlen, maxlen = int(sys.argv[3]), int(sys.argv[4])
    user_ids = np.loadtxt(infile).astype(int)
    distinct_user_id = np.unique(user_ids)
    user_rows = []
    
    for uid in distinct_user_id:
        user_rows.append( np.where(user_ids == uid)[0] )

    # print(user_rows)
    
    splits = []
    for user_row in user_rows:
        for seq in split(user_row, minlen, maxlen): 
            # these are continuous sequence of indices
            # saving start and end would suffice
            splits.append([seq[0], seq[-1]])
    
    splits = np.array(splits)
    splits = np.insert(splits, 0, 0, axis=1)

    for i in range(len(splits)):
        if user_ids[splits[i][1]] != user_ids[splits[i][2]]:
            splits[i][0] = -1
        else:
            splits[i][0] = user_ids[splits[i][1]]

    np.savetxt(outfile, splits, fmt='%i', delimiter=",")

except Exception as e:
    print (e)

