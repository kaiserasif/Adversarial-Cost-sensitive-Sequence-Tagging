Hard-code validation indices file in **validation_set.txt** the training data directory required. The file contains sequence indices, in a shuffled order. 5-fold cross validation performed with these samples, without shuffling, to ensure same splits across the two models. 

Python 2 is requierd for SVM.

## To run Adversarial Sequence Tagging:
Run in python 3, with cvxopt installed. 

`PYTHONPATH=<adv_par> python hapt_experiment/run_experiment.py <data_dir> <cost_file> <svm_data_save_dir>`

- **adv_par**: path to directory containing AdversarialGame library
- **data_dir**: contains **Train/{X_train.txt,sequnce_splits_train.txt} Test/{X_test.txt,sequnce_splits_test.txt}**
- **cost_file**: csv file containing the cost matrix
- **svm_data_save_dir**: (_optional_) if provided, svm style features saved


## To run struc-svm

Use the run<span/>.sh as in **Python 2**:

`run.sh <data_dir> <cost_file>`

- **data_dir** : contains **train.dat**, **test.dat** and **validation_set.txt**
- **cost_file** : csv file containing the cost matrix
