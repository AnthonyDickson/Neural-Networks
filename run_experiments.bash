rm -rf results/
mkdir results
python experiments.py --n-jobs ${1:-1} 2> errors.log
