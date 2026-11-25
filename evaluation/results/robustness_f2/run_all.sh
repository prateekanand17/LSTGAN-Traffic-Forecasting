#!/usr/bin/env bash
# F2 robustness grid (no API). One process per forecaster seed; experiments run in order within a seed.
cd "$(dirname "$0")/../../.."
seed=$1; shift
# one BLAS/OpenMP thread per process: several processes run in parallel on small arrays
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for e in "$@"; do python evaluation/run_robustness.py --experiment "$e" --seed "$seed" > "evaluation/results/robustness_f2/log_${e}_s${seed}.txt" 2>&1 || echo "FAILED $e s$seed"; done
echo "seed $seed finished"
