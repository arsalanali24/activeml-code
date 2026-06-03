#!/bin/bash
#SBATCH --job-name=fe2s2_dmrg
#SBATCH --partition=largemem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120GB
#SBATCH --time=12:00:00
#SBATCH --output=/pc2/users/h/hpcmual/activeml/logs/fe2s2_dmrg_%j.out
#SBATCH --error=/pc2/users/h/hpcmual/activeml/logs/fe2s2_dmrg_%j.err

module purge
module load lang
module load Python/3.11.3-GCCcore-12.3.0
export PATH=$HOME/.local/bin:$PATH
export OMP_NUM_THREADS=8
export LD_PRELOAD=/pc2/users/h/hpcmual/activeml/lib/lapack_wrapper.so
export LD_LIBRARY_PATH=/opt/software/pc2/EB-SW/software/OpenBLAS/0.3.23-GCC-12.3.0/lib:$LD_LIBRARY_PATH

echo "Started at $(date)"
echo "Node: $SLURMD_NODENAME"
python3 -u ~/activeml/scripts/test_fe2s2_dmrg.py
echo "Finished at $(date)"
