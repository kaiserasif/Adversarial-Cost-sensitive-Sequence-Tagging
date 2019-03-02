"""A module for SVM^python for multiclass learning."""

# Thomas Finley, tfinley@gmail.com

import svmapi
import ast, sys
import numpy as np

# during trainng sparm.cost_matrix can be created
# but classifier fails to accept such parameters
cost_matrix = [[0, 1], [1, 0]]

def parse_parameters(sparm):
    """Sets attributes of sparm based on command line arguments.
    
    This gives the user code a chance to change sparm based on the
    custom command line arguments.  The custom command line arguments
    are stored in sparm.argv as a list of strings.  The custom command
    lines are stored in '--option', then 'value' sequence.
    
    If this function is not implemented, any custom command line
    arguments are ignored and sparm remains unchanged."""
    # sparm.arbitrary_parameter = 'I am an arbitrary parameter!'
    
    # kaiser 190227
    if (sparm.argv) : 
        for idx in range(0, len(sparm.argv), 2):
            parse_parameters_classify(sparm.argv[idx], sparm.argv[idx+1])
            # sparm.cost_matrix = ast.literal_eval(sparm.argv[i+1])

def parse_parameters_classify(attribute, value):
    global cost_matrix
    if '--cost' == attribute:
        cost_matrix = ast.literal_eval(value) # cannot write to svmapi.Sparm
    if '--cost_file' == attribute:
        cost_matrix = np.loadtxt(value, delimiter=',')
        print (cost_matrix)


def read_examples(filename, sparm):
    """Parses an input file into an example sequence."""
    # This reads example files of the type read by SVM^multiclass.
    examples = []
    x, y = [], []
    # Open the file and read each example.
    for line in file(filename):
        # Get rid of comments.
        if line.find('#'): line = line[:line.find('#')]
        tokens = line.split()
        # If the line is empty, one sequence ended
        if not tokens: 
            if x: examples.append((x, y))
            x, y = [], []
            continue
        # Get the target.
        target = int(tokens[0])
        # Get the features.
        tokens = [t.split(':') for t in tokens[1:]]
        features = [(0,1)]+[(int(k),float(v)) for k,v in tokens]
        # Add the example to the list
        x.append(svmapi.Sparse(features))
        y.append(target)
    if x: # non added sample of the end
        examples.append((x, y))
    # Print out some very useful statistics.
    # print (examples)
    print len(examples),'examples read in multiclass_seq.py'
    return examples


def init_model(sample, sm, sparm):
    """Store the number of features and classes in the model."""
    # Note that these features will be stored in the model and written
    # when it comes time to write the model to a file, and restored in
    # the classifier when reading the model from the file.
    sm.num_features = max(max(xi) for x,y in sample for xi in x)[0]+1
    sm.num_classes = max(max(y) for x,y in sample)
    sm.size_psi = (sm.num_features * sm.num_classes # n_feature for n_classe
                + sm.num_classes * sm.num_classes) # n_class * n_class transistion 
    #print 'size_psi set to',sm.size_psi
    # fix the cost_matrix
    # if not hasattr(sparm, 'cost_matrix') or len(sparm.cost_matrix) != sm.num_classes:
    global cost_matrix
    if len(cost_matrix) < sm.num_classes:
        cost_matrix = [[1] * sm.num_classes for _ in range(sm.num_classes)]
        for i in range(sm.num_classes):
            cost_matrix[i][i] = 0

    print 'init_model'


def viterbi(x, y, sm, sparm):
    """
    Do viterby to find the maximum 
    score y_bar, or maximum violating 
    constraint depending on y is None or list
    """
    # do viterbi
    # add loss(y, y_bar) to each node, if y present
    # sm.w has first n_feature weights for each of n_class
    # then there are the transition weights of n_class x n_class
    # print 'viterbi'
    n_class, n_feature = sm.num_classes, sm.num_features
    T = len(x) 
    y_bar = [0] * T
    score = [[0.] * n_class for _ in range(T) ]
    history = [[1<<31] * n_class for _ in range(T) ] # for backtracking
    # print x
    for t in range(T):
        # the node features
        for idx,v in x[t]:
            # if not 0 <= idx < n_feature: continue # weird issue with very high key values
            for c in range(n_class):
                score[t][c] += sm.w[c * n_feature + idx] * v
        # add the loss if y provided
        if y is not None:
            score[t][c] += cost_matrix[c][y[t]-1] # 0 based index
        # search the transition path
        # for all prev y, search max of score + y-y transistion
        # 00 01 02 10 11 12 20 21 22
        # index = Offset + (n_class) * prev_step + cur_step # 0 based index
        if t > 0:
            M = n_class * n_feature
            for c in range(n_class):
                prev_max, prev_max_c = score[t-1][0] + sm.w[M + 0 + c], 0
                for prev_c in range(1, n_class):
                    tmp = score[t-1][prev_c] + sm.w[M + prev_c * n_class + c]
                    if tmp > prev_max:
                        prev_max, prev_max_c = tmp, prev_c
                score[t][c] += prev_max
                history[t][c] = prev_max_c

    # now find the maximum score in the end state
    # and backtrack the history to retrieve path
    # values saved 0-based index, y are 1 based
    max_score = max(score[-1])
    y_bar[-1] = score[-1].index(max_score) + 1
    for t in range(T-1, 0, -1):
        y_bar[t-1] = history[t][y_bar[t]-1] + 1
    return y_bar, max_score



def classify_example(x, sm, sparm):
    # print 'classify_example'
    """Returns the classification of an example 'x'."""
    # # Construct the discriminant-label pairs.
    # scores = [(classification_score(x,c,sm,sparm), c)
    #           for c in xrange(1,sm.num_classes+1)]
    # # Return the label with the max discriminant value.
    # print list(sm.w)
    y_bar, score = viterbi(x, None, sm, sparm)
    return y_bar # max(scores)[1]


def find_most_violated_constraint(x, y, sm, sparm):
    # print 'find_most_violated_constraint'
    """Returns the most violated constraint for example (x,y)."""
    # # Similar, but include the loss.
    # scores = [(classification_score(x,c,sm,sparm)+loss(y,c,sparm), c)
    #           for c in xrange(1,sm.num_classes+1)]
    # ybar = max(scores)[1]
    # #print y, ybar
    y_bar, score = viterbi(x, y, sm, sparm)
    # print y, y_bar
    return y_bar # max(scores)[1]


def psi(x, y, sm, sparm):
    # print 'psi'
    """Returns the combined feature vector Psi(x,y)."""
    T = len(y)
    n_class, n_feature = sm.num_classes, sm.num_features
    ps = [0] * sm.size_psi
    edge_offset = n_feature * n_class
    for t in range(T):
        offset = n_feature * (y[t]-1)
        for k,v in x[t]: ps[k+offset] += v
        if t > 0: ps[edge_offset + (y[t-1]-1)*n_class + y[t]-1 ] += 1
        
    return svmapi.Sparse(ps)


def loss(y, ybar, sparm):
    """Use the sparm.cost_matrix to compute loss"""
    loss = 0
    # print (len(y), len(ybar))
    for i in xrange(len(y)):
        loss += cost_matrix[ybar[i]-1][y[i]-1] # 0 based index
    # print 'loss', loss
    return loss


def write_label(fileptr, y):
    print>>fileptr, ",".join(str(i) for i in y)


def eval_prediction(exnum, (x, y), ypred, sm, sparm, teststats):
    """Accumulate statistics about a single training example.
    
    Allows accumulated statistics regarding how well the predicted
    label ypred for pattern x matches the true label y.  The first
    time this function is called teststats is None.  This function's
    return value will be passed along to the next call to
    eval_prediction.  After all test predictions are made, the last
    value returned will be passed along to print_testing_stats.

    On the first call, that is, when exnum==0, teststats==None.  The
    default behavior is that the function does nothing."""
    if exnum==0: teststats = []
    # print 'on example',exnum,'predicted',ypred,'where correct is',y
    teststats.append((loss(y, ypred, sparm), len(y)))
    # print (teststats)
    return teststats


def print_testing_stats(sample, sm, sparm, teststats):
    total_loss, total_length = 0.0, 0

    with open('test_prediction_stats.txt', 'wt') as f:
        for loss, length in teststats:
            total_loss += loss
            total_length += length
            f.write("%d,%f,%f\n" % (length, loss, 1.0 *loss / length) )

    print '%d'%len(teststats), 'sequences, micro-average error:', (total_loss / total_length)
