#!/bin/bash

# directory structure of the codes
#  run.sh
#  run.py
#  svm-python-v204/svm_multiclass_learn
#  svm-python-v204/svm_multiclass_classify
#  svm-python-v204/multiclass_seq.py

usage="run.sh <data_dir> <cost_file>"
if [ $# -lt 2 ]; then
    echo $usage
    exit 1
fi
data_dir=$1
cost_file=$2
DIR=$( dirname "${BASH_SOURCE[0]}" )
echo $DIR
python "$DIR"/run.py "$DIR"/svm-python-v204/svm_python_learn "$DIR"/svm-python-v204/svm_python_classify "$data_dir" "$cost_file"
