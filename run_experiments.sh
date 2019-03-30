#!/usr/bin/env bash

# Argument parsing adapted from https://stackoverflow.com/a/5476278
usage="$(basename "$0") [-h] [-t n] [-j n] [-s n][-e] -- Utility for running experiments with grid search.

where:
    -h  show this help text
    -t N_TRIALS     How many times each grid search parameter set should be run (default: 20)
    -j N_JOBS       The number of processors to use (default: 1)
    -s N_SPLITS     How many folds to use for cross validations (default: 5)
    -e SEND_EMAIL   Flag indicating that an email should be sent when all grid search runs are finished.
                    Make sure you have the environment variable 'EMAIL_NOTIFICATION_ADDRESS' set to the
                    email address you want to receive the email notification with.
                    This email is likely to end up in your spam folder."

n_trials=20
n_jobs=1
n_splits=5
send_email=false

while getopts ':ht:j::e' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
    t) n_trials=$OPTARG
       ;;
    j) n_jobs=$OPTARG
       ;;
    s) n_splits=$OPTARG
       ;;
    e) send_email=true
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

date

bash clean.sh

sep="********************************************************************************"

echo "$sep"
echo "Generating Configuration Files..."
python generate_grid_search_cfgs.py
cfgs=($(ls | grep .json))
echo "$sep"
echo

trap "echo && exit" INT

for cfg in "${cfgs[@]}"
do
    echo "$sep"
    echo "Running Grid Search for Configuration File: $cfg"
    echo "Number of Jobs: $n_jobs"
    echo "Number of Trials per Configuration: $n_trials"
    echo
    python grid_search.py ${cfg} --n-jobs ${n_jobs} --n-trials ${n_trials} --n-splits ${n_splits} 2>> errors.log
    echo "$sep"
    echo
done

if [[ ${send_email} = true ]]; then
    echo "" | mail -s "Grid Search has Finished Training." ${EMAIL_NOTIFICATION_ADDRESS}
fi
