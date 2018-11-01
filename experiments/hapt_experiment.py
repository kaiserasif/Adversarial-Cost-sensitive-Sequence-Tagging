import numpy as np
from sklearn import metrics

from AdversarialGame.classifiers import CostSensitiveClassifier, CostSensitiveSequenceTagger
from analysis.dataloader import load_hapt_data

def run():
    # load data, scale may not be needed, get number of classes, make start at 0
    X_tr, y_tr, X_ts, y_ts = load_hapt_data()
    # find n_class and min_y
    ys = np.unique( np.concatenate(y_tr + y_ts) )
    min_y = min(ys)
    n_class = max(ys) - min_y + 1
    for y in y_tr: y -= min_y
    for y in y_ts: y -= min_y

    # now create classifier and train
    cost_matrix = 1 - np.eye(n_class)
    ast = CostSensitiveSequenceTagger(cost_matrix=cost_matrix, max_itr=100)
    ast.fit(X_tr, y_tr)

    # predict
    y_pred = ast.predict(X_ts)
    for y, yp in zip(y_ts, y_pred):
        print(len(y), metrics.accuracy_score(y, yp))
    

if __name__ == '__main__':
    run()