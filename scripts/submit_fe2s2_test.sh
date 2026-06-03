#!/bin/bash
#SBATCH --job-name=fe2s2_test
#SBATCH --partition=largemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=02:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fe2s2_test_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fe2s2_test_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8

echo "Started at $(date)"
python3 ~/activeml/scripts/test_fe2s2_bs.py
echo "Finished at $(date)"
