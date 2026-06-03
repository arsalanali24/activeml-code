#!/bin/bash
#SBATCH --job-name=fe2s2_single
#SBATCH --partition=largemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60GB
#SBATCH --time=01:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fe2s2_single_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fe2s2_single_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8
export LD_PRELOAD=/pc2/users/h/hpcmual/activeml/lib/lapack_wrapper.so
export LD_LIBRARY_PATH=/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.23-GCC-12.3.0/lib:$LD_LIBRARY_PATH

TESTFILE=$(head -1 /pc2/users/h/hpcmual/activeml/scripts/fe2s2_commands.txt)
echo "Testing: $TESTFILE"
echo "Started: $(date)"
python3 ~/activeml/scripts/fe2s2_features.py $TESTFILE
echo "Finished: $(date)"
