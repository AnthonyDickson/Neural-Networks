rm -rf results/
mkdir results
python experiments.py --n-jobs ${1:-1} --n-trials ${2:-20} 2> errors.log
