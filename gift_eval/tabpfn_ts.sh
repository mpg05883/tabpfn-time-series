#!/bin/bash

# Delta partition information:
# https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html#delta-partitions-queues

#SBATCH --job-name=tabpfn_eval
#SBATCH --partition=gpuA40x4  
#SBATCH --array=0-96
#SBATCH --mem=100GB  # Total memory per node
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2  # * Ensure --gpus-per-node equals --ntasks-per-node
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest
#SBATCH --account=bcqc-delta-gpu
#SBATCH --time=12:00:00
#SBATCH --output=output/logs/%x/out/%A/%a.out
#SBATCH --error=output/logs/%x/err/%A/%a.err
#SBATCH --mail-user=mpgee@usc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p ./output/logs
source utils.sh
log_info "Starting ${SLURM_JOB_NAME}. Job ID: ${SLURM_JOB_ID}"

# Activate conda environment
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate gift

if python evaluate.py; then
    log_info "Successfully finished ${SLURM_JOB_NAME}!"
    log_error "No errors!"
    
    done_file=$(get_done_file)
    end_time=$(get_timestamp)
    echo "[${end_time}] Done with ${SLURM_JOB_NAME}!" >"$done_file"
else
    log_error "ERROR: ${SLURM_JOB_NAME} failed!"
    exit 1
fi